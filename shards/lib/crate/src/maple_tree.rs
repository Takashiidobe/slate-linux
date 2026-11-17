
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
//! Maple Tree implementation
//! Copyright (c) 2018-2022 Oracle Corporation
//! Authors: Liam R. Howlett <liam@infradead.org>
//! 	    Matthew Wilcox <willy@infradead.org>
//! Copyright (c) 2023 ByteDance
//! Author: Peng Zhang <zhangpeng.00@bytedance.com>
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
//! Yes, this permits 64-bit accesses on 32-bit architectures. These will
//! actually be atomic in some cases (namely Armv7 + LPAE), but for others we
//! rely on the access being split into 2x32-bit accesses for a 32-bit quantity
//! (e.g. a virtual address) and a strong prevailing wind.
//! # chars in a path name including nul
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
//! Create a dummy function pointer reference to prevent objtool from marking
//! the function as needing to be "sealed" (i.e. ENDBR converted to NOP by
//! apply_seal_endbr()).
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
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
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! Sign Flag
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
//! 64bit FS base
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
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! LIST_HEAD_INIT - initialize a &struct list_head's links to point to itself
//! @name: name of the list_head
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! Wrap the arch provided IRQ routines to provide appropriate checks.
//! New user namespace
//! Ensure that leaf 0x2 cache and TLB type values do not intersect,
//! since they share the same type field at struct cpuid_0x2_table.
//! virt_to_page(kaddr) returns a valid pointer if and only if
//! virt_addr_valid(kaddr) returns true.
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
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
//! These constants are defined in the MSan LLVM instrumentation pass.
//! ignore signal
//! bit-flags
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
//! Convenience macros for the sake of set_current_state:
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
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! The trampoline is 8 bytes and of the general form:
//!
//!   jmp.d32 \func
//!   ud1 %esp, %ecx
//!
//! That trailing #UD provides both a speculation stop and serves as a unique
//! 3 byte signature identifying static call trampolines. Also see tramp_ud[]
//! and __static_call_fixup().
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! These are specified by iBCS2
//! Epoll event masks
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! The PERF_METRICS MSR is modeled as several magic fixed-mode PMCs, one for
//! each TopDown metric event.
//!
//! Internally the TopDown metric events are mapped to the FxCtr 3 (SLOTS).
//! Macro to enforce the same ordering and stack sizes
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! This is a list of *what* is being read, not *how* nor *where*.
//! ld/ldx fields
//! Define a list of socket types which can be the argument for
//! skc_to_*_sock() helpers. All these sockets should have
//! sock_common as the first argument in its memory layout.
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! 31 pointers + header align the folio_batch structure to a power of two
//! Choose 31 as it makes rqspinlock_held cacheline-aligned.
//! Used for event string fields when they are NULL
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
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// flex_array_size() - Calculate size of a flexible array member
//                     within an enclosing structure.
// @p: Pointer to the structure.
// @member: Name of the flexible array member.
// @count: Number of elements in the array.
//
// Calculates size of a flexible array of @count number of @member
// elements, at the end of structure @p.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
// struct_size() - Calculate size of structure with trailing flexible array.
// @p: Pointer to the structure.
// @member: Name of the array member.
// @count: Number of elements in the array.
//
// Calculates size of memory needed for structure of @p followed by an
// array of @count number of @member elements.
//
// Return: number of bytes needed or SIZE_MAX on overflow.
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
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Concatenate two parameters, but allow them to be expanded beforehand.
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
// Xen paravirtual guest
// Intel-defined CPU features, CPUID level 0x00000007:0 (EBX), word 9
// "fsgsbase" RDFSBASE, WRFSBASE, RDGSBASE, WRGSBASE instructions
// "erms" Enhanced REP MOVSB/STOSB instructions
// "smap" Supervisor Mode Access Prevention
// "clflushopt" CLFLUSHOPT instruction
// "clwb" CLWB instruction
// IA32_TSC_DEADLINE and X2APIC MSRs need fencing
// "lass" Linear Address Space Separation
// Fast short REP STOSB
// "fred" Flexible Return and Event Delivery
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
// No such file or directory
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Permission denied
// Bad address
// Device or resource busy
// File exists
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Math result not representable
// Operation not supported on transport endpoint
// Memory page has hardware error
// Overflow Flag
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No static inline type checking - see Subtlety (1) above.
// SCHED_ISO: reserved but not implemented yet
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
// disable sas during sighandling
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
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
// Same as rb_first(), but O(1)
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// PT_TRACE_* event enable flags
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
__ALIGN_KERNEL__2!($x, $a)
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
macro_rules! IS_ENABLED {
() => {
__is_defined!()
};
}

/// Special values for ma_state.node.
/// MA_ERROR represents an errno.  After dropping the lock and attempting
/// to resolve the error, the walk would have to be restarted from the
/// top of the tree as the tree may have been modified.
macro_rules! MA_ERROR {
($err:expr) => {
((($err) as u64) << 2i32 | 2) as *mut maple_enode
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

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u64).wrapping_sub(1))
};
}

macro_rules! __SLAB_FLAG_BIT {
($nr:expr) => {
1u32 << $nr
};
}

macro_rules! __WRITE_ONCE__15 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut (), ($val) as *mut ()) };
if true {
break;
}
}
};
}

macro_rules! __WRITE_ONCE {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut *mut (), $val) };
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
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__19 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__11 {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__24 {
() => {
((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__25 {
() => {
((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

macro_rules! __preempt_count_inc {
() => {
__preempt_count_add(1i32);
};
}

macro_rules! __rcu_dereference_protected__9 {
($p:expr, $c:expr) => {
{
let mut __v: *mut ();
RCU_LOCKDEP_WARN!(!$c);
__v = $p;
__v
}
};
}

macro_rules! __rcu_dereference_protected__2 {
($p:expr, $c:expr) => {
{
let mut __v: *mut ();
RCU_LOCKDEP_WARN!(($c) == 0);
__v = $p;
__v
}
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

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__5 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<__SlateRecord_types_h_197>() as *const ();
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
macro_rules! __verify_pcpu_ptr__6 {
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

/// __verify_pcpu_ptr() verifies @ptr is a percpu pointer without evaluating
/// @ptr and is invoked once before a percpu area is accessed by all
/// accessors and operations.  This is performed in the generic part of
/// percpu and arch overrides don't need to worry about it; however, if an
/// arch wants to implement an arch-specific percpu accessor or operation,
/// it may use __verify_pcpu_ptr() to verify the parameters.
///
/// + 0 is required in order to convert the pointer type from a
/// potential array type to a pointer to a single item of the array.
macro_rules! __verify_pcpu_ptr__7 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<hlist_head>() as *const ();
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

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__21 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut $type;
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

macro_rules! kcsan_wmb {
() => {
while false {
}
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

macro_rules! likely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! likely__2 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! likely__8 {
($x:expr) => {
!$x.is_null() as i64
};
}

macro_rules! likely__5 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! likely__4 {
($x:expr) => {
!$x.is_null() as i64
};
}

macro_rules! ma_enode_ptr {
($x:expr) => {
($x) as *mut maple_enode
};
}

macro_rules! ma_mnode_ptr {
($x:expr) => {
$x
};
}

macro_rules! ma_mnode_ptr__2 {
($x:expr) => {
($x) as *mut maple_node
};
}

macro_rules! ma_parent_ptr {
($x:expr) => {
($x) as *mut maple_pnode
};
}

macro_rules! ma_parent_ptr__2 {
($x:expr) => {
$x
};
}

macro_rules! mas_lock {
($mas:expr) => {
spin_lock(unsafe { core::ptr::addr_of_mut!((*(*$mas).tree).__slate_anon_0.ma_lock) });
};
}

macro_rules! mas_lock_nested {
($mas:expr, $subclass:expr) => {
spin_lock_nested!(unsafe { core::ptr::addr_of_mut!((*(*$mas).tree).__slate_anon_0.ma_lock) }, $subclass);
};
}

macro_rules! mas_tree_parent {
($x:expr) => {
((unsafe { (*$x).tree }) as u64) | 1
};
}

macro_rules! mas_unlock {
($mas:expr) => {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*(*$mas).tree).__slate_anon_0.ma_lock) });
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

macro_rules! might_sleep_if {
($cond:expr) => {
loop {
if $cond {
might_sleep!();
}
if true {
break;
}
}
};
}

macro_rules! mt_lock_is_held {
() => {
1i32
};
}

macro_rules! mt_slot_count {
($x:expr) => {
unsafe { mt_slots[(mte_node_type(($x) as *const maple_enode) as u32) as usize] }
};
}

macro_rules! mt_write_lock_is_held {
() => {
1i32
};
}

macro_rules! mtree_lock {
($mt:expr) => {
spin_lock(unsafe { core::ptr::addr_of_mut!((*$mt).__slate_anon_0.ma_lock) });
};
}

macro_rules! mtree_unlock {
($mt:expr) => {
spin_unlock(unsafe { core::ptr::addr_of_mut!((*$mt).__slate_anon_0.ma_lock) });
};
}

macro_rules! offsetof__6 {
() => {
8u64
};
}

/// We abuse bit 3 from flags to pass exact information, see
/// perf_arch_misc_flags() and the comment with PERF_EFLAGS_EXACT.
macro_rules! perf_arch_fetch_caller_regs {
($regs:expr, $__ip:expr) => {
unsafe {
(*$regs).ip = $__ip;
}
unsafe {
(*$regs).sp = __slate_intrinsic_2d7c29df8fdf6378(const { ((0 as i32) as u32) as i32 }) as u64;
}
unsafe {
(*$regs).__slate_anon_17.cs = 16u16;
}
unsafe {
(*$regs).flags = 0;
}
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
};
}

/// Always evaluate the 'subclass' argument to avoid that the compiler
/// warns about set-but-not-used variables when building with
/// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
macro_rules! raw_spin_lock_nested {
($lock:expr, $subclass:expr) => {
$subclass;
unsafe { _raw_spin_lock($lock) };
};
}

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
};
}

/// rcu_dereference_protected() - fetch RCU pointer when updates prevented
/// @p: The pointer to read, prior to dereferencing
/// @c: The conditions under which the dereference will take place
///
/// Return the value of the specified RCU-protected pointer, but omit
/// the READ_ONCE().  This is useful in cases where update-side locks
/// prevent the value of the pointer from changing.  Please note that this
/// primitive does *not* prevent the compiler from repeating this reference
/// or combining it with other references, so it should not be used without
/// protection of appropriate locks.
///
/// This function is only for update-side use.  Using this function
/// when protected only by rcu_read_lock() will result in infrequent
/// but very ugly failures.
macro_rules! rcu_dereference_protected {
($p:expr, $c:expr) => {
__rcu_dereference_protected__2!($p, $c)
};
}

/// rcu_dereference_protected() - fetch RCU pointer when updates prevented
/// @p: The pointer to read, prior to dereferencing
/// @c: The conditions under which the dereference will take place
///
/// Return the value of the specified RCU-protected pointer, but omit
/// the READ_ONCE().  This is useful in cases where update-side locks
/// prevent the value of the pointer from changing.  Please note that this
/// primitive does *not* prevent the compiler from repeating this reference
/// or combining it with other references, so it should not be used without
/// protection of appropriate locks.
///
/// This function is only for update-side use.  Using this function
/// when protected only by rcu_read_lock() will result in infrequent
/// but very ugly failures.
macro_rules! rcu_dereference_protected__5 {
($p:expr, $c:expr) => {
__rcu_dereference_protected__9!($p, $c)
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

macro_rules! spin_lock_nested {
($lock:expr, $subclass:expr) => {
loop {
raw_spin_lock_nested!(spinlock_check($lock), $subclass);
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

macro_rules! test_bit {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if false && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*$addr) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! unlikely__6 {
($x:expr) => {
!$x.is_null() as i64
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

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
};
}

unsafe extern "C" {
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __SCK__might_resched: static_call_key;
    static mut this_cpu_off: u64;
    static mut __cpu_online_mask: cpumask;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule_notrace: static_call_key;
    static mut cpu_number: i32;
    static mut rcu_callback_map: lockdep_map;
    static mut tracepoint_srcu: srcu_struct;
    static mut kstat: kernel_stat;
    static mut perf_callchain_entry: perf_callchain_entry;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn ____wrong_branch_error() -> bool;
    fn __bad_size_call_parameter();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __compiletime_assert_27();
    fn __compiletime_assert_54();
    fn lock_is_held(arg0: *const ()) -> i32;
    fn lockdep_is_held(arg0: *const ()) -> i32;
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn call_rcu(arg0: *mut callback_head, arg1: Option<unsafe extern "C" fn(*mut callback_head)>);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn kvfree_call_rcu(arg0: *mut kvfree_rcu_head, arg1: *mut ());
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn __kmem_cache_create_args(arg0: *const u8, arg1: u32, arg2: *mut kmem_cache_args, arg3: u32) -> *mut kmem_cache;
    fn kfree(arg0: *const ());
    fn kmem_cache_alloc_noprof(arg0: *mut kmem_cache, arg1: u32) -> *mut ();
    fn kmem_cache_free_bulk(arg0: *mut kmem_cache, arg1: u64, arg2: *mut *mut ());
    fn kmem_cache_prefill_sheaf(arg0: *mut kmem_cache, arg1: u32, arg2: u32) -> *mut slab_sheaf;
    fn kmem_cache_refill_sheaf(arg0: *mut kmem_cache, arg1: u32, arg2: *mut *mut slab_sheaf, arg3: u32) -> i32;
    fn kmem_cache_return_sheaf(arg0: *mut kmem_cache, arg1: u32, arg2: *mut slab_sheaf);
    fn kmem_cache_alloc_from_sheaf_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: *mut slab_sheaf) -> *mut ();
    fn kmem_cache_sheaf_size(arg0: *mut slab_sheaf) -> u32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_400();
    fn __compiletime_assert_421();
    fn __SCT__tp_func_ma_op(arg0: *mut (), arg1: *const u8, arg2: *mut ma_state) -> i32;
    fn __compiletime_assert_470();
    fn __SCT__tp_func_ma_read(arg0: *mut (), arg1: *const u8, arg2: *mut ma_state) -> i32;
    fn __compiletime_assert_477();
    fn __SCT__tp_func_ma_write(arg0: *mut (), arg1: *const u8, arg2: *mut ma_state, arg3: u64, arg4: *mut ()) -> i32;
    fn __compiletime_assert_484();
    fn __compiletime_assert_488();
    fn __compiletime_assert_489();
    fn __compiletime_assert_491();
    fn __compiletime_assert_492();
    fn __compiletime_assert_494();
    fn __compiletime_assert_495();
    fn trace_raw_output_prep(arg0: *mut trace_iterator, arg1: *mut trace_event) -> i32;
    fn trace_event_printf(arg0: *mut trace_iterator, arg1: *const u8, ...);
    fn trace_handle_return(arg0: *mut trace_seq) -> u32;
    fn trace_event_reg(arg0: *mut trace_event_call, arg1: u32, arg2: *mut ()) -> i32;
    fn trace_event_buffer_reserve(arg0: *mut trace_event_buffer, arg1: *mut trace_event_file, arg2: u64) -> *mut ();
    fn trace_event_buffer_commit(arg0: *mut trace_event_buffer);
    fn __compiletime_assert_683();
    fn __trace_trigger_soft_disabled(arg0: *mut trace_event_file) -> bool;
    fn trace_event_raw_init(arg0: *mut trace_event_call) -> i32;
    fn perf_trace_buf_alloc(arg0: i32, arg1: *mut *mut pt_regs, arg2: *mut i32) -> *mut ();
    fn perf_trace_run_bpf_submit(arg0: *mut (), arg1: i32, arg2: i32, arg3: *mut trace_event_call, arg4: u64, arg5: *mut pt_regs, arg6: *mut hlist_head, arg7: *mut task_struct);
    fn __compiletime_assert_692();
    fn __builtin_unreachable();
    fn __compiletime_assert_694();
    fn __compiletime_assert_699();
    fn __compiletime_assert_701();
    fn __compiletime_assert_707();
    fn __compiletime_assert_708();
    fn __compiletime_assert_709();
    fn __compiletime_assert_710();
    fn __compiletime_assert_711();
    fn __compiletime_assert_712();
    fn __compiletime_assert_719();
    fn __compiletime_assert_720();
    fn __compiletime_assert_721();
    fn __compiletime_assert_726();
    fn __compiletime_assert_727();
    fn __compiletime_assert_728();
    fn __compiletime_assert_729();
    fn __compiletime_assert_730();
    fn __compiletime_assert_731();
    fn __compiletime_assert_732();
    fn __compiletime_assert_733();
    fn __compiletime_assert_734();
    fn __compiletime_assert_735();
    fn __compiletime_assert_736();
    fn __compiletime_assert_737();
    fn __compiletime_assert_738();
    fn __compiletime_assert_739();
    fn __compiletime_assert_740();
    fn __compiletime_assert_743();
    fn __compiletime_assert_744();
    fn __compiletime_assert_745();
    fn __compiletime_assert_746();
    fn __compiletime_assert_747();
    fn __compiletime_assert_748();
    fn __compiletime_assert_749();
    fn __compiletime_assert_750();
    fn __compiletime_assert_751();
    fn __compiletime_assert_752();
    fn __compiletime_assert_753();
    fn __compiletime_assert_754();
    fn __compiletime_assert_755();
    fn __compiletime_assert_756();
    fn __compiletime_assert_757();
    fn __compiletime_assert_758();
    fn __compiletime_assert_759();
    fn __compiletime_assert_760();
    fn __compiletime_assert_761();
    fn __compiletime_assert_762();
    fn __compiletime_assert_763();
    fn __compiletime_assert_766();
    fn __compiletime_assert_767();
    fn __compiletime_assert_768();
    fn __compiletime_assert_769();
    fn __compiletime_assert_770();
    fn __compiletime_assert_771();
    fn __compiletime_assert_772();
    fn __compiletime_assert_773();
    fn __compiletime_assert_774();
    fn __compiletime_assert_775();
    fn __compiletime_assert_776();
    fn __compiletime_assert_777();
    fn __compiletime_assert_778();
    fn __compiletime_assert_779();
    fn __compiletime_assert_780();
    fn __compiletime_assert_781();
    fn __compiletime_assert_782();
    fn __compiletime_assert_783();
    fn __compiletime_assert_784();
    fn __compiletime_assert_785();
    fn __compiletime_assert_786();
    fn __compiletime_assert_792();
    fn __compiletime_assert_800();
    fn __compiletime_assert_803();
    fn __compiletime_assert_807();
    fn __compiletime_assert_808();
    fn __compiletime_assert_809();
    fn __compiletime_assert_850();
    fn __compiletime_assert_851();
    fn __compiletime_assert_852();
    fn __compiletime_assert_853();
    fn __compiletime_assert_856();
    fn __compiletime_assert_857();
    fn __compiletime_assert_858();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.frameaddress.p0"]
    fn __slate_intrinsic_2d7c29df8fdf6378(arg0: i32) -> *mut ();
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
    #[link_name = "llvm.returnaddress.p0"]
    fn __slate_intrinsic_ded315013ea6523(arg0: i32) -> *mut ();
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
struct kvfree_rcu_head {
    next: *mut kvfree_rcu_head,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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
struct lock_class_key {
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
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
core::arch::asm!("2: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 2b - ., 3; .popsection\n# {0}", const 2 as i32, options(nostack, att_syntax));
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
core::arch::asm!("3: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 3b - ., 4; .popsection\n# {0}", const 3 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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

#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_4: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ma_op, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ma_read, const 0 as u8, label {
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
fn arch_static_branch__const3(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ma_write, const 0 as u8, label {
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits103U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
union __SlateRecord_ptrace_h_144 {
    cs: u16,
    csx: u64,
    fred_cs: fred_cs,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_ptrace_h_156 {
    ss: u16,
    ssx: u64,
    fred_ss: fred_ss,
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
fn hlist_empty(h: *const hlist_head) -> i32 {
let mut __v0: *mut hlist_node;
loop {
if !(__native_word__25!() || ((const { core::mem::size_of::<*mut hlist_node>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_27() };
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
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
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
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
}

// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

// to_cpumask - convert a NR_CPUS bitmap to a struct cpumask *
// @bitmap: the bitmap
//
// There are a few places where cpumask_var_t isn't appropriate and
// static cpumasks must be used (eg. very early boot), yet we don't
// expose the definition of 'struct cpumask'.
//
// This does the conversion, and can be used as a constant initializer.
#[inline(always)]
fn cpu_online(cpu: u32) -> bool {
cpumask_test_cpu(cpu as i32, (unsafe { core::ptr::addr_of_mut!(__cpu_online_mask) }) as *const cpumask)
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct cacheline_padding {
    x: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_54() };
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
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits135U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits136U0,
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
struct pt_regs {
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    bp: u64,
    bx: u64,
    r11: u64,
    r10: u64,
    r9: u64,
    r8: u64,
    ax: u64,
    cx: u64,
    dx: u64,
    si: u64,
    di: u64,
    orig_ax: u64,
    ip: u64,
    __slate_anon_17: __SlateRecord_ptrace_h_144,
    flags: u64,
    sp: u64,
    __slate_anon_20: __SlateRecord_ptrace_h_156,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits123U0,
    pkru: u32,
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
// We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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

#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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
struct __SlateRecord_preempt_h_497 {
    lock: *mut (),
}


#[inline(always)]
fn class_preempt_notrace_destructor(_T: *mut __SlateRecord_preempt_h_497) {
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely!(__preempt_count_dec_and_test()) != 0 {
loop {
unsafe {
core::arch::asm!("call __SCT__preempt_schedule_notrace", options(att_syntax));
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
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_notrace_59: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule_notrace) }) as *mut ();

#[inline(always)]
fn class_preempt_notrace_constructor() -> __SlateRecord_preempt_h_497 {
let _t: __SlateRecord_preempt_h_497 = __SlateRecord_preempt_h_497 { lock: 1i32 as *mut () };
loop {
__preempt_count_inc!();
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct sysv_shm {
    shm_clist: list_head,
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
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
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
struct net_device {
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
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
struct k_sigaction {
    sa: sigaction,
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
    __slate_bits_0: __slate_bits::__SlateBits228U0,
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
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
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
struct bpf_local_storage {
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
struct posix_cputimers {
    bases: [posix_cputimer_base; 3],
    timers_active: u32,
    expiry_active: u32,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct gendisk {
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
struct maple_pnode {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_103 {
    slot: [*mut (); 16],
    __slate_anon_1: __SlateRecord_maple_tree_h_105,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_105 {
    pad: [*mut (); 15],
    meta: maple_metadata,
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
    __slate_bits_0: __slate_bits::__SlateBits70U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits70U1,
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
    __slate_bits_2: __slate_bits::__SlateBits70U2,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_enode {
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
struct __SlateRecord_maple_tree_h_162 {
    node: *mut maple_node,
    max: u64,
    mt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_167 {
    node: *mut maple_node,
    max: u64,
    start: u8,
    end: u8,
    mt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_178 {
    pivot: [u64; 3],
    __slate_anon_1: __SlateRecord_maple_tree_h_180,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_180 {
    _pad: [*mut (); 2],
    max: u64,
}


fn rcu_read_lock_held() -> i32 {
1
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_324 {
    __slate_anon_0: __SlateRecord_maple_tree_h_325,
    __slate_anon_1: __SlateRecord_maple_tree_h_329,
    mr64: maple_range_64,
    ma64: maple_arange_64,
    cp: maple_copy,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_325 {
    parent: *mut maple_pnode,
    slot: [*mut (); 31],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_329 {
    pad: *mut (),
    rcu: callback_head,
    piv_parent: *mut maple_enode,
    parent_slot: u8,
    r#type: u32,
    slot_len: u8,
    ma_flags: u32,
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

// RCU_INIT_POINTER() - initialize an RCU protected pointer
// @p: The pointer to be initialized.
// @v: The value to initialized the pointer to.
//
// Initialize an RCU-protected pointer in special cases where readers
// do not need ordering constraints on the CPU or the compiler.  These
// special cases are:
//
// 1.	This use of RCU_INIT_POINTER() is NULLing out the pointer *or*
// 2.	The caller has taken whatever steps are required to prevent
// 	RCU readers from concurrently accessing this pointer *or*
// 3.	The referenced data structure has already been exposed to
// 	readers either at compile time or via rcu_assign_pointer() *and*
//
// 	a.	You have not made *any* reader-visible changes to
// 		this structure since then *or*
// 	b.	It is OK for readers accessing this structure from its
// 		new location to see the old state of the structure.  (For
// 		example, the changes were to statistical counters or to
// 		other state where exact synchronization is not required.)
//
// Failure to follow these rules governing use of RCU_INIT_POINTER() will
// result in impossible-to-diagnose memory corruption.  As in the structures
// will look OK in crash dumps, but any concurrent RCU readers might
// see pre-initialized values of the referenced data structure.  So
// please be very careful how you use RCU_INIT_POINTER()!!!
//
// If you are creating an RCU-protected linked structure that is accessed
// by a single external-to-structure RCU-protected pointer, then you may
// use RCU_INIT_POINTER() to initialize the internal RCU-protected
// pointers, but you must use rcu_assign_pointer() to initialize the
// external-to-structure pointer *after* you have completely initialized
// the reader-accessible portions of the linked structure.
//
// Note that unlike rcu_assign_pointer(), RCU_INIT_POINTER() provides no
// ordering guarantees for either the CPU or the compiler.
// kfree_rcu() - kfree an object after a grace period.
// @ptr: pointer to kfree for double-argument invocations.
// @rhf: the name of the struct rcu_head within the type of @ptr.
//
// Many rcu callbacks functions just call kfree() on the base structure.
// These functions are trivial, but their size adds up, and furthermore
// when they are used in a kernel module, that module must invoke the
// high-latency rcu_barrier() function at module-unload time.
//
// The kfree_rcu() function handles this issue. In order to have a universal
// callback function handling different offsets of rcu_head, the callback needs
// to determine the starting address of the freed object, which can be a large
// kmalloc or vmalloc allocation. To allow simply aligning the pointer down to
// page boundary for those, only offsets up to 4095 bytes can be accommodated.
// If the offset is larger than 4095 bytes, a compile-time error will
// be generated in kvfree_rcu_arg_2(). If this error is triggered, you can
// either fall back to use of call_rcu() or rearrange the structure to
// position the rcu_head structure into the first 4096 bytes.
//
// The object to be freed can be allocated either by kmalloc(),
// kmalloc_nolock(), or kmem_cache_alloc().
//
// Note that the allowable offset might decrease in the future.
//
// The BUILD_BUG_ON check must not involve any function calls, hence the
// checks are done in macros here.
// The BUILD_BUG_ON() makes sure the rcu_head offset can be handled. See the
// comment of kfree_rcu() for details.
#[repr(C)]
#[derive(Clone, Copy)]
struct slab_sheaf {
}


// 256 bytes including ->parent
// 256 bytes
// 240 bytes
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_metadata {
    end: u8,
    gap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_range_64 {
    parent: *mut maple_pnode,
    pivot: [u64; 15],
    __slate_anon_2: __SlateRecord_maple_tree_h_103,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_arange_64 {
    parent: *mut maple_pnode,
    pivot: [u64; 9],
    slot: [*mut (); 10],
    gap: [u64; 10],
    meta: maple_metadata,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_topiary {
    parent: *mut maple_pnode,
    next: *mut maple_enode,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_copy {
    dst: [__SlateRecord_maple_tree_h_162; 3],
    src: [__SlateRecord_maple_tree_h_167; 4],
    slot: [*mut (); 3],
    gap: [u64; 3],
    min: u64,
    __slate_anon_5: __SlateRecord_maple_tree_h_178,
    end: u8,
    s_count: u8,
    d_count: u8,
    split: u8,
    data: u8,
    height: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


// DOC: Maple tree flags
//
// * MT_FLAGS_ALLOC_RANGE	- Track gaps in this tree
// * MT_FLAGS_USE_RCU		- Operate in RCU mode
// * MT_FLAGS_HEIGHT_OFFSET	- The position of the tree height in the flags
// * MT_FLAGS_HEIGHT_MASK	- The mask for the maple tree height value
// * MT_FLAGS_LOCK_MASK		- How the mt_lock is used
// * MT_FLAGS_LOCK_IRQ		- Acquired irq-safe
// * MT_FLAGS_LOCK_BH		- Acquired bh-safe
// * MT_FLAGS_LOCK_EXTERN	- mt_lock is not used
//
// MAPLE_HEIGHT_MAX	The largest height that can be stored
#[repr(C)]
#[derive(Clone, Copy)]
struct maple_tree {
    __slate_anon_0: __SlateRecord_maple_tree_h_261,
    ma_flags: u32,
    ma_root: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct maple_node {
    __slate_anon_0: __SlateRecord_maple_tree_h_324,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ma_topiary {
    head: *mut maple_enode,
    tail: *mut maple_enode,
    mtree: *mut maple_tree,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.__mt_destroy")]
extern "C" fn __mt_destroy(mt: *mut maple_tree) {
let root: *mut () = mt_root_locked(mt);
let _r_a_p__v: u64 = 0;
if true && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_856() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*mt).ma_root, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_857() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_858() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { core::ptr::addr_of_mut!((*mt).ma_root) }, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
if xa_is_node(root as *const ()) {
mte_destroy_walk(root as *mut maple_enode, mt);
}
unsafe {
(*mt).ma_flags = mt_attr(mt);
}
}

fn mtree_empty(mt: *const maple_tree) -> bool {
unsafe { (*mt).ma_root }.is_null()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ma_state {
    tree: *mut maple_tree,
    index: u64,
    last: u64,
    node: *mut maple_enode,
    min: u64,
    max: u64,
    sheaf: *mut slab_sheaf,
    alloc: *mut maple_node,
    node_request: u64,
    status: u32,
    depth: u8,
    offset: u8,
    mas_flags: u8,
    end: u8,
    store_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ma_wr_state {
    mas: *mut ma_state,
    node: *mut maple_node,
    r_min: u64,
    r_max: u64,
    r#type: u32,
    offset_end: u8,
    pivots: *mut u64,
    end_piv: u64,
    slots: *mut *mut (),
    entry: *mut (),
    content: *mut (),
    vacant_height: u8,
    sufficient_height: u8,
}


// When changing MA_STATE, remember to also change rust/kernel/maple_tree.rs
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_alloc_cyclic")]
extern "C" fn mas_alloc_cyclic(mas: *mut ma_state, startp: *mut u64, entry: *mut (), mut range_lo: u64, range_hi: u64, next: *mut u64, gfp: u32) -> i32 {
let mut __UNIQUE_ID_y__791: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__790: u64 = unsafe { core::mem::zeroed() };
let mut min: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u64 = core::ptr::addr_of_mut!(min) as *mut u64;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_9 = range_lo;
'__join_0: {
loop {
__UNIQUE_ID_x__790 = *__slate_slot_9;
__UNIQUE_ID_y__791 = *next;
while false {
}
range_lo = if __UNIQUE_ID_x__790 > __UNIQUE_ID_y__791 { __UNIQUE_ID_x__790 } else { __UNIQUE_ID_y__791 };
*__slate_slot_10 = mas_empty_area(mas, range_lo, range_hi, 1u64);
if *__slate_slot_10 < 0 && range_lo > *__slate_slot_9 {
mas_reset(mas);
*__slate_slot_10 = mas_empty_area(mas, *__slate_slot_9, range_hi, 1u64);
if *__slate_slot_10 == 0 {
*__slate_slot_10 = 1;
}
}
if *__slate_slot_10 < 0 {
break '__join_0;
} else {
mas_insert(mas, entry);
if !mas_nomem(mas, gfp) {
break;
}
}
}
if mas_is_err(mas) {
*__slate_slot_10 = xa_err((*mas).node as *mut ());
} else {
if (*(*mas).tree).ma_flags & 2048 != 0 && *__slate_slot_10 == 0 {
let __v5: *mut maple_tree = (*mas).tree;
let __v3: u32 = (*__v5).ma_flags & 4294965247u32;
(*__v5).ma_flags = __v3;
*__slate_slot_10 = 1;
}
*startp = (*mas).index;
*next = unsafe { *startp }.wrapping_add(1);
if *next == 0 {
let __v2: *mut maple_tree = (*mas).tree;
(*__v2).ma_flags = (*__v2).ma_flags | 2048;
}
}
}
mas_destroy(mas);
return *__slate_slot_10;
}
}

fn mas_is_active(mas: *mut ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 0
}

fn mas_is_err(mas: *mut ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 7
}

#[inline(always)]
fn mas_reset(mas: *mut ma_state) {
unsafe {
(*mas).status = 1;
}
unsafe {
(*mas).node = core::ptr::null_mut::<maple_enode>();
}
}

fn __mas_set_range(mas: *mut ma_state, start: u64, last: u64) {
// Ensure the range starts within the current slot
let mut __v0: i64;
let __ret_warn_on: i32 = (mas_is_active(mas) && ((unsafe { (*mas).index }) > start || (unsafe { (*mas).last }) < start)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("110: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 110b - ., 3; .popsection\n# {0}", const 110 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___mas_set_range_1, sym __slate_asm_string___mas_set_range_2, const 797 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("111: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 111b - ., 4; .popsection\n# {0}", const 111 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
(*mas).index = start;
}
unsafe {
(*mas).last = last;
}
}

fn mas_set_range(mas: *mut ma_state, start: u64, last: u64) {
mas_reset(mas);
__mas_set_range(mas, start, last);
}

fn mas_set(mas: *mut ma_state, index: u64) {
mas_set_range(mas, index, index);
}

fn mt_external_lock(mt: *const maple_tree) -> bool {
(unsafe { (*mt).ma_flags }) & 768 == 768
}

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


fn mt_in_rcu(mt: *mut maple_tree) -> bool {
(unsafe { (*mt).ma_flags }) & 2 != 0
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


fn mt_height(mt: *const maple_tree) -> u32 {
((unsafe { (*mt).ma_flags }) & 124) >> 2i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_nulls_node {
    next: *mut hlist_nulls_node,
    pprev: *mut *mut hlist_nulls_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_queue_head {
    lock: spinlock,
    head: list_head,
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
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct completion {
    done: u32,
    wait: swait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct alloc_tag {
    ct: codetag,
    counters: *mut alloc_tag_counters,
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


fn mem_alloc_profiling_enabled() -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
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


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__19!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_167() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*ssp).srcu_ctrp) as *const *mut srcu_ctr) };
scp = __v0;
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__6!();
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
} else {
let mut __v13: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__5!();
let mut __v15: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v16: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v16 = pfo_val__;
tcp_ptr__ = __v16;
let __v18: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_locks) }) as u64);
tcp_ptr__ = __v18;
__v15 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v13 = __v15;
atomic_long_inc(__v13);
}
// Y, and implicit RCU reader.
// Y, and implicit RCU reader.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
__acquire_shared!();
scp
}

fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: i64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
core::arch::asm!("incw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
core::arch::asm!("incl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
core::arch::asm!("incq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: i64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (core::ptr::addr_of_mut!((*scp).srcu_unlocks.counter) as u64) as *mut i64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
} else {
let mut __v12: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__5!();
let mut __v14: *mut __SlateRecord_types_h_197;
let mut tcp_ptr__: u64 = 0;
let mut __v15: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v15 = pfo_val__;
tcp_ptr__ = __v15;
let __v17: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_unlocks) }) as u64);
tcp_ptr__ = __v17;
__v14 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v12 = __v14;
atomic_long_inc(__v12);
}
// Z, and implicit RCU reader.
// Z, and implicit RCU reader.
}

#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
}

// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
fn srcu_read_lock_fast_notrace(ssp: *mut srcu_struct) -> *mut srcu_ctr {
srcu_check_read_flavor(ssp, 4i32);
let retval: *mut srcu_ctr = __srcu_read_lock_fast(ssp);
retval
}

fn srcu_read_unlock_fast_notrace(ssp: *mut srcu_struct, scp: *mut srcu_ctr) {
srcu_check_read_flavor(ssp, 4i32);
__srcu_read_unlock_fast(ssp, scp);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
}

#[inline(always)]
fn class_srcu_fast_notrace_constructor(l: *mut srcu_struct) -> __SlateRecord_srcu_h_645 {
let mut _t: __SlateRecord_srcu_h_645 = {
let mut __t0: __SlateRecord_srcu_h_645 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_srcu_h_645 = core::ptr::addr_of_mut!(_t);
unsafe {
(*_T).scp = srcu_read_lock_fast_notrace((*_T).lock);
}
return _t;
}

#[inline(always)]
fn __class_srcu_fast_notrace_cleanup_ctx(_T: *mut *mut __SlateRecord_srcu_h_645) {
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
struct uprobes_state {
    xol_area: *mut xol_area,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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
    __slate_bits_0: __slate_bits::__SlateBits446U0,
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
fn gfpflags_allow_blocking(gfp_flags: u32) -> bool {
gfp_flags & (BIT!(10i32) as u32) != 0
}

// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
fn fs_reclaim_acquire(gfp_mask: u32) {
}

fn fs_reclaim_release(gfp_mask: u32) {
}

fn might_alloc(gfp_mask: u32) {
fs_reclaim_acquire(gfp_mask);
fs_reclaim_release(gfp_mask);
if (unsafe { (*get_current()).flags }) & 2048 != 0 {
return;
}
might_sleep_if!(gfpflags_allow_blocking(gfp_mask));
}

#[repr(C)]
#[derive(Clone, Copy)]
struct wake_irq {
}


fn xa_mk_internal(v: u64) -> *mut () {
(v << 2i32 | 2) as *mut ()
}

fn xa_is_internal(entry: *const ()) -> bool {
(entry as u64) & 3 == 2
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


fn xa_is_zero(entry: *const ()) -> bool {
unlikely!(entry == (xa_mk_internal(257u64) as *const ())) != 0
}

fn xa_is_err(entry: *const ()) -> bool {
let __v0: bool = xa_is_internal(entry) && entry >= (xa_mk_internal(4095i32.wrapping_neg() as u64) as *const ());
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fwnode_handle {
}


fn xa_err(entry: *mut ()) -> i32 {
// xa_to_internal() would not do sign extension.
if xa_is_err(entry as *const ()) {
return ((entry as i64) >> 2i32) as i32;
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


fn xa_is_node(entry: *const ()) -> bool {
xa_is_internal(entry) && (entry as u64) > 4096
}

fn xa_is_advanced(entry: *const ()) -> bool {
xa_is_internal(entry) && entry <= (xa_mk_internal(256u64) as *const ())
}

// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits479U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


// Panic if kmem_cache_create() fails
#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache_args {
    align: u32,
    useroffset: u32,
    usersize: u32,
    freeptr_offset: u32,
    use_freeptr_offset: bool,
    ctor: Option<unsafe extern "C" fn(*mut ())>,
    sheaf_capacity: u32,
}


fn __kmem_cache_create(name: *const u8, size: u32, align: u32, flags: u32, ctor: Option<unsafe extern "C" fn(*mut ())>) -> *mut kmem_cache {
let mut kmem_args: kmem_cache_args = {
let mut __t0: kmem_cache_args = unsafe { core::mem::zeroed() };
__t0.align = align;
__t0.ctor = ctor;
__t0
};
unsafe { __kmem_cache_create_args(name, size, core::ptr::addr_of_mut!(kmem_args), flags) }
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


fn __kmem_cache_default_args(name: *const u8, size: u32, args: *mut kmem_cache_args, flags: u32) -> *mut kmem_cache {
let mut kmem_default_args: kmem_cache_args = {
let mut __t0: kmem_cache_args = unsafe { core::mem::zeroed() };
__t0
};
// Make sure we don't get passed garbage.
let mut __v0: i64;
let __ret_warn_on: i32 = !args.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("218: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 218b - ., 3; .popsection\n# {0}", const 218 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string___kmem_cache_default_args_1, sym __slate_asm_string___kmem_cache_default_args_2, const 448 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("219: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 219b - ., 4; .popsection\n# {0}", const 219 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return ERR_PTR(22i32.wrapping_neg() as i64) as *mut kmem_cache;
}
unsafe { __kmem_cache_create_args(name, size, core::ptr::addr_of_mut!(kmem_default_args), flags) }
}

// kmem_cache_create - Create a kmem cache.
// @__name: A string which is used in /proc/slabinfo to identify this cache.
// @__object_size: The size of objects to be created in this cache.
// @__args: Optional arguments, see &struct kmem_cache_args. Passing %NULL
// 	    means defaults will be used for all the arguments.
//
// This is currently implemented as a macro using ``_Generic()`` to call
// either the new variant of the function, or a legacy one.
//
// The new variant has 4 parameters:
// ``kmem_cache_create(name, object_size, args, flags)``
//
// See __kmem_cache_create_args() which implements this.
//
// The legacy variant has 5 parameters:
// ``kmem_cache_create(name, object_size, align, flags, ctor)``
//
// The align and ctor parameters map to the respective fields of
// &struct kmem_cache_args
//
// Context: Cannot be called within a interrupt, but can be interrupted.
//
// Return: a pointer to the cache on success, NULL on failure.
// no-op
// kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
// kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
// and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
// __alloc_objs - Allocate objects of a given type using
// @KMALLOC: which size-based kmalloc wrapper to allocate with.
// @GFP: GFP flags for the allocation.
// @TYPE: type to allocate space for.
// @COUNT: how many @TYPE objects to allocate.
//
// Returns: Newly allocated pointer to (first) @TYPE of @COUNT-many
// allocated @TYPE objects, or NULL on failure.
// All kzalloc aliases for kmalloc_(obj|objs|flex).
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
struct folio_queue {
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
    __slate_bits_0: __slate_bits::__SlateBits501U0,
    vm_ops: *const vm_operations_struct,
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
    __slate_bits_0: __slate_bits::__SlateBits503U0,
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
struct mnt_idmap {
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
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
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
struct key_type {
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits528U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits528U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits528U2,
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
    __slate_bits_3: __slate_bits::__SlateBits528U3,
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
struct tty_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_dir {
    header: ctl_table_header,
    root: rb_root,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tty_audit_buf {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ctl_table_set {
    is_seen: Option<unsafe extern "C" fn(*mut ctl_table_set) -> i32>,
    dir: ctl_dir,
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
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhlist_head {
    rhead: rhash_head,
    next: *mut rhlist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct iovec {
    iov_base: *mut (),
    iov_len: u64,
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
struct fserror_event {
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
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


fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpu_itimer {
    expires: u64,
    incr: u64,
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
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
    __slate_bits_0: __slate_bits::__SlateBits296U0,
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


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_400() };
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct lockref {
    __slate_anon_0: __SlateRecord_lockref_h_26,
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
struct path {
    mnt: *mut vfsmount,
    dentry: *mut dentry,
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
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
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
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ring_buffer_iter {
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
if !(__native_word__6!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_421() };
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
union __SlateRecord_perf_event_h_410 {
    sample_period: u64,
    sample_freq: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_471 {
    wakeup_events: u32,
    wakeup_watermark: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_477 {
    bp_addr: u64,
    kprobe_func: u64,
    uprobe_path: u64,
    config1: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_483 {
    bp_len: u64,
    kprobe_addr: u64,
    probe_offset: u64,
    config2: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_530 {
    aux_action: u32,
    __slate_anon_1: __SlateRecord_perf_event_h_532,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_532 {
    __slate_bits_0: __slate_bits::__SlateBits790U0,
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
struct __SlateRecord_perf_event_h_1322 {
    __slate_bits_0: __slate_bits::__SlateBits792U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1512 {
    var1_dw: u32,
    var2_w: u16,
    var3_w: u16,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uts_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ipc_namespace {
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
struct time_namespace {
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
struct proc_ns_operations {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_ns_common_types_h_112 {
    __ns_ref: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_pin {
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
struct ftrace_ops {
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
struct cgroup_taskset {
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
struct __SlateRecord_bpf_h_1528 {
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_flags: u32,
    inner_map_fd: u32,
    numa_node: u32,
    map_name: [u8; 16],
    map_ifindex: u32,
    btf_fd: u32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    map_extra: u64,
    value_type_btf_obj_fd: i32,
    map_token_fd: i32,
    excl_prog_hash: u64,
    excl_prog_hash_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1581 {
    map_fd: u32,
    key: u64,
    __slate_anon_2: __SlateRecord_bpf_h_1584,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1584 {
    value: u64,
    next_key: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1591 {
    in_batch: u64,
    out_batch: u64,
    keys: u64,
    values: u64,
    count: u32,
    map_fd: u32,
    elem_flags: u64,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1608 {
    prog_type: u32,
    insn_cnt: u32,
    insns: u64,
    license: u64,
    log_level: u32,
    log_size: u32,
    log_buf: u64,
    kern_version: u32,
    prog_flags: u32,
    prog_name: [u8; 16],
    prog_ifindex: u32,
    expected_attach_type: u32,
    prog_btf_fd: u32,
    func_info_rec_size: u32,
    func_info: u64,
    func_info_cnt: u32,
    line_info_rec_size: u32,
    line_info: u64,
    line_info_cnt: u32,
    attach_btf_id: u32,
    __slate_anon_20: __SlateRecord_bpf_h_1633,
    core_relo_cnt: u32,
    fd_array: u64,
    core_relos: u64,
    core_relo_rec_size: u32,
    log_true_size: u32,
    prog_token_fd: i32,
    fd_array_cnt: u32,
    signature: u64,
    signature_size: u32,
    keyring_id: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1633 {
    attach_prog_fd: u32,
    attach_btf_obj_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1674 {
    pathname: u64,
    bpf_fd: u32,
    file_flags: u32,
    path_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1687 {
    __slate_anon_0: __SlateRecord_bpf_h_1688,
    attach_bpf_fd: u32,
    attach_type: u32,
    attach_flags: u32,
    replace_bpf_fd: u32,
    __slate_anon_5: __SlateRecord_bpf_h_1696,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1688 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1696 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1703 {
    prog_fd: u32,
    retval: u32,
    data_size_in: u32,
    data_size_out: u32,
    data_in: u64,
    data_out: u64,
    repeat: u32,
    duration: u32,
    ctx_size_in: u32,
    ctx_size_out: u32,
    ctx_in: u64,
    ctx_out: u64,
    flags: u32,
    cpu: u32,
    batch_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1727 {
    __slate_anon_0: __SlateRecord_bpf_h_1728,
    next_id: u32,
    open_flags: u32,
    fd_by_id_token_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1728 {
    start_id: u32,
    prog_id: u32,
    map_id: u32,
    btf_id: u32,
    link_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1740 {
    bpf_fd: u32,
    info_len: u32,
    info: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1746 {
    __slate_anon_0: __SlateRecord_bpf_h_1747,
    attach_type: u32,
    query_flags: u32,
    attach_flags: u32,
    prog_ids: u64,
    __slate_anon_5: __SlateRecord_bpf_h_1755,
    __slate_bits_0: __slate_bits::__SlateBits862U0,
    prog_attach_flags: u64,
    link_ids: u64,
    link_attach_flags: u64,
    revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1747 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1755 {
    prog_cnt: u32,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1769 {
    name: u64,
    prog_fd: u32,
    __slate_bits_0: __slate_bits::__SlateBits865U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1776 {
    btf: u64,
    btf_log_buf: u64,
    btf_size: u32,
    btf_log_size: u32,
    btf_log_level: u32,
    btf_log_true_size: u32,
    btf_flags: u32,
    btf_token_fd: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1794 {
    pid: u32,
    fd: u32,
    flags: u32,
    buf_len: u32,
    buf: u64,
    prog_id: u32,
    fd_type: u32,
    probe_offset: u64,
    probe_addr: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1810 {
    __slate_anon_0: __SlateRecord_bpf_h_1811,
    __slate_anon_1: __SlateRecord_bpf_h_1815,
    attach_type: u32,
    flags: u32,
    __slate_anon_4: __SlateRecord_bpf_h_1821,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1811 {
    prog_fd: u32,
    map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1815 {
    target_fd: u32,
    target_ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1821 {
    target_btf_id: u32,
    __slate_anon_1: __SlateRecord_bpf_h_1823,
    perf_event: __SlateRecord_bpf_h_1827,
    kprobe_multi: __SlateRecord_bpf_h_1834,
    tracing: __SlateRecord_bpf_h_1841,
    netfilter: __SlateRecord_bpf_h_1850,
    tcx: __SlateRecord_bpf_h_1856,
    uprobe_multi: __SlateRecord_bpf_h_1863,
    netkit: __SlateRecord_bpf_h_1873,
    cgroup: __SlateRecord_bpf_h_1880,
    tracing_multi: __SlateRecord_bpf_h_1887,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1823 {
    iter_info: u64,
    iter_info_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1827 {
    bpf_cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1834 {
    flags: u32,
    cnt: u32,
    syms: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1841 {
    target_btf_id: u32,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1850 {
    pf: u32,
    hooknum: u32,
    priority: i32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1856 {
    __slate_anon_0: __SlateRecord_bpf_h_1857,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1857 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1863 {
    path: u64,
    offsets: u64,
    ref_ctr_offsets: u64,
    cookies: u64,
    cnt: u32,
    flags: u32,
    pid: u32,
    path_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1873 {
    __slate_anon_0: __SlateRecord_bpf_h_1874,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1874 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1880 {
    __slate_anon_0: __SlateRecord_bpf_h_1881,
    expected_revision: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1881 {
    relative_fd: u32,
    relative_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1887 {
    ids: u64,
    cookies: u64,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1895 {
    link_fd: u32,
    __slate_anon_1: __SlateRecord_bpf_h_1897,
    flags: u32,
    __slate_anon_3: __SlateRecord_bpf_h_1904,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1897 {
    new_prog_fd: u32,
    new_map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1904 {
    old_prog_fd: u32,
    old_map_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1916 {
    link_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1920 {
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1924 {
    link_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1929 {
    prog_fd: u32,
    map_fd: u32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1935 {
    flags: u32,
    bpffs_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1940 {
    stream_buf: u64,
    stream_buf_len: u32,
    stream_id: u32,
    prog_fd: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1947 {
    map_fd: u32,
    prog_fd: u32,
    flags: u32,
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
union __SlateRecord_bpf_h_6842 {
    raw_tracepoint: __SlateRecord_bpf_h_6843,
    tracing: __SlateRecord_bpf_h_6849,
    cgroup: __SlateRecord_bpf_h_6856,
    iter: __SlateRecord_bpf_h_6860,
    netns: __SlateRecord_bpf_h_6884,
    xdp: __SlateRecord_bpf_h_6888,
    struct_ops: __SlateRecord_bpf_h_6891,
    netfilter: __SlateRecord_bpf_h_6894,
    kprobe_multi: __SlateRecord_bpf_h_6900,
    uprobe_multi: __SlateRecord_bpf_h_6907,
    tracing_multi: __SlateRecord_bpf_h_6917,
    perf_event: __SlateRecord_bpf_h_6926,
    tcx: __SlateRecord_bpf_h_6959,
    netkit: __SlateRecord_bpf_h_6963,
    sockmap: __SlateRecord_bpf_h_6967,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6843 {
    tp_name: u64,
    tp_name_len: u32,
    __slate_bits_0: __slate_bits::__SlateBits898U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6849 {
    attach_type: u32,
    target_obj_id: u32,
    target_btf_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits899U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6856 {
    cgroup_id: u64,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6860 {
    target_name: u64,
    target_name_len: u32,
    __slate_anon_2: __SlateRecord_bpf_h_6868,
    __slate_anon_3: __SlateRecord_bpf_h_6873,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6868 {
    map: __SlateRecord_bpf_h_6869,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6869 {
    map_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6873 {
    cgroup: __SlateRecord_bpf_h_6874,
    task: __SlateRecord_bpf_h_6878,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6874 {
    cgroup_id: u64,
    order: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6878 {
    tid: u32,
    pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6884 {
    netns_ino: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6888 {
    ifindex: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6891 {
    map_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6894 {
    pf: u32,
    hooknum: u32,
    priority: i32,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6900 {
    addrs: u64,
    count: u32,
    flags: u32,
    missed: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6907 {
    path: u64,
    offsets: u64,
    ref_ctr_offsets: u64,
    cookies: u64,
    path_size: u32,
    count: u32,
    flags: u32,
    pid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6917 {
    attach_type: u32,
    count: u32,
    btf_obj_id: u32,
    __slate_bits_0: __slate_bits::__SlateBits913U0,
    ids: u64,
    addrs: u64,
    cookies: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6926 {
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits914U0,
    __slate_anon_2: __SlateRecord_bpf_h_6929,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_6929 {
    uprobe: __SlateRecord_bpf_h_6930,
    kprobe: __SlateRecord_bpf_h_6937,
    tracepoint: __SlateRecord_bpf_h_6945,
    event: __SlateRecord_bpf_h_6951,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6930 {
    file_name: u64,
    name_len: u32,
    offset: u32,
    cookie: u64,
    ref_ctr_offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6937 {
    func_name: u64,
    name_len: u32,
    offset: u32,
    addr: u64,
    missed: u64,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6945 {
    tp_name: u64,
    name_len: u32,
    __slate_bits_0: __slate_bits::__SlateBits918U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6951 {
    config: u64,
    r#type: u32,
    __slate_bits_0: __slate_bits::__SlateBits919U0,
    cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6959 {
    ifindex: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6963 {
    ifindex: u32,
    attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_6967 {
    map_id: u32,
    attach_type: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_btf_h_64 {
    size: u32,
    r#type: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct btf {
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
struct bpf_verifier_log {
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
struct bpf_verifier_env {
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
struct bpf_arena {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_iter_aux_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_local_storage_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_state {
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
union __SlateRecord_bpf_h_256 {
    kptr: btf_field_kptr,
    graph_root: btf_field_graph_root,
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
    __slate_bits_0: __slate_bits::__SlateBits525U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_329 {
    work: work_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_offload_dev {
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
union __SlateRecord_bpf_h_996 {
    __slate_anon_0: __SlateRecord_bpf_h_997,
    arg_type: [u32; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_997 {
    arg1_type: u32,
    arg2_type: u32,
    arg3_type: u32,
    arg4_type: u32,
    arg5_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1006 {
    __slate_anon_0: __SlateRecord_bpf_h_1007,
    arg_btf_id: [*mut u32; 12],
    __slate_anon_2: __SlateRecord_bpf_h_1015,
    arg_size: [u64; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1007 {
    arg1_btf_id: *mut u32,
    arg2_btf_id: *mut u32,
    arg3_btf_id: *mut u32,
    arg4_btf_id: *mut u32,
    arg5_btf_id: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1015 {
    arg1_size: u64,
    arg2_size: u64,
    arg3_size: u64,
    arg4_size: u64,
    arg5_size: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
    system: *const u8,
    eval_string: *const u8,
    eval_value: u64,
}


// Declare an exported function that Rust code can call to trigger this
// tracepoint. This function does not include the static branch; that is done
// in Rust to avoid a function call when the tracepoint is disabled.
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
// We have no guarantee that gcc and the linker won't up-align the tracepoint
// structures, so we create an array of pointers that will be used for iteration
// on the tracepoints.
//
// it_func[0] is never NULL because there is at least one element in the array
// when the array itself is non NULL.
// 									 * Annotate the probestub 'CFI_NOSEAL' to stop objtool from		 * requesting the kernel remove the ENDBR, because the only		 * references to the function are in the __tracepoint section,		 * that objtool doesn't scan.
// tracepoint_string - register constant persistent string to trace system
// @str - a constant persistent string that will be referenced in tracepoints
//
// If constant strings are being used in tracepoints, it is faster and
// more efficient to just save the pointer to the string and reference
// that with a printf "%s" instead of saving the string in the ring buffer
// and wasting space and time.
//
// The problem with the above approach is that userspace tools that read
// the binary output of the trace buffers do not have access to the string.
// Instead they just show the address of the string which is not very
// useful to users.
//
// With tracepoint_string(), the string will be registered to the tracing
// system and exported to userspace via the debugfs/tracing/printk_formats
// file that maps the string address to the string text. This way userspace
// tools that read the binary buffers have a way to map the pointers to
// the ASCII strings they represent.
//
// The @str used must be a constant string and persistent as it would not
// make sense to show a string that no longer exists. But it is still fine
// to be used with modules, because when modules are unloaded, if they
// had tracepoints, the ring buffers are cleared too. As long as the string
// does not change during the life of the module, it is fine to use
// tracepoint_string() within a module.
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1109 {
    ctx_field_size: i32,
    __slate_anon_1: __SlateRecord_bpf_h_1111,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1111 {
    btf: *mut btf,
    btf_id: u32,
    ref_id: u32,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.__traceiter_ma_op")]
extern "C" fn __traceiter_ma_op(mut __data: *mut (), r#fn: *const u8, mas: *mut ma_state) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_488() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ma_op)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_487: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_487;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_489() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state)>>(it_func) }.unwrap()(__data, r#fn, mas) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_reg_state {
}


fn __do_trace_ma_op(r#fn: *const u8, mas: *mut ma_state) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
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
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_467: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_467) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_468: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_468) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_470() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ma_op)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_469: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_469;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>>(__SCT__tp_func_ma_op as *const ()) };
unsafe { __v12.unwrap()(__data, r#fn, mas) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ma_op: [u8; 6] = [109, 97, 95, 111, 112, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ma_op_471: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ma_op) }) as *mut ();

fn trace_ma_op(r#fn: *const u8, mas: *mut ma_state) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ma_op.key) }) != 0 {
__do_trace_ma_op(r#fn, mas);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_472: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.__traceiter_ma_read")]
extern "C" fn __traceiter_ma_read(mut __data: *mut (), r#fn: *const u8, mas: *mut ma_state) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_491() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ma_read)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_490: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_490;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_492() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state)>>(it_func) }.unwrap()(__data, r#fn, mas) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

fn __do_trace_ma_read(r#fn: *const u8, mas: *mut ma_state) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
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
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_474: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_474) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_475: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_475) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_477() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ma_read)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_476: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_476;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>>(__SCT__tp_func_ma_read as *const ()) };
unsafe { __v12.unwrap()(__data, r#fn, mas) };
}
if true {
break;
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1393 {
    rcu: callback_head,
    work: work_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1423 {
    model: btf_func_model,
    addr: *mut (),
    ftrace_managed: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1442 {
    old_image: *mut bpf_tramp_image,
    old_flags: u32,
}


#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ma_read: [u8; 8] = [109, 97, 95, 114, 101, 97, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ma_read_478: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ma_read) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1705 {
    tail_call: __SlateRecord_bpf_h_1706,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1706 {
    map: *mut bpf_map,
    key: u32,
}


fn trace_ma_read(r#fn: *const u8, mas: *mut ma_state) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ma_read.key) }) != 0 {
__do_trace_ma_read(r#fn, mas);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_479: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_kfunc_desc_tab {
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.__traceiter_ma_write")]
extern "C" fn __traceiter_ma_write(mut __data: *mut (), r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) -> i32 {
let mut it_func_ptr: *mut tracepoint_func = unsafe { core::mem::zeroed() };
let mut __v0: *mut tracepoint_func;
let mut __v1: *mut tracepoint_func;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_494() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ma_write)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_493: *mut tracepoint_func = __v1;
__v0 = __UNIQUE_ID_rcu_493;
it_func_ptr = __v0;
if !it_func_ptr.is_null() {
loop {
let mut __v3: *mut ();
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_495() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*it_func_ptr).func) as *const *mut ()) };
let it_func: *mut () = __v3;
__data = unsafe { (*it_func_ptr).data };
unsafe { unsafe { core::mem::transmute::<*mut (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ())>>(it_func) }.unwrap()(__data, r#fn, mas, piv, val) };
let __v6: *mut tracepoint_func = it_func_ptr;
let __v7: *mut tracepoint_func = __v6.wrapping_offset(1 as isize);
it_func_ptr = __v7;
if unsafe { (*__v7).func }.is_null() {
break;
}
}
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1821_2 {
    keyring_serial: i32,
    keyring_type: u8,
    verdict: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_kfunc_btf_tab {
}


fn __do_trace_ma_write(r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) {
let mut __v0: i32;
let mut pscr_ret__: i32 = 0;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<i32>() as *const ();
if true {
break;
}
}
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __v2: i32;
let mut pfo_val__: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movb %gs:{1}(%rip), {0:l}", lateout(reg) __t0, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movw %gs:{1}(%rip), {0:x}", lateout(reg) __t1, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movl %gs:{1}(%rip), {0:e}", lateout(reg) __t2, sym cpu_number, options(nostack, att_syntax));
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
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t3, sym cpu_number, options(nostack, att_syntax));
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
if cpu_online(__v0 as u32) {
let mut __UNIQUE_ID_guard_481: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_481) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_482: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_482) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__11!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_484() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ma_write)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_483: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_483;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ()) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ()) -> i32>>(__SCT__tp_func_ma_write as *const ()) };
unsafe { __v12.unwrap()(__data, r#fn, mas, piv, val) };
}
if true {
break;
}
}
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1892 {
    work: work_struct,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1926 {
    digest: [u8; 32],
    tag: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock_fprog_kern {
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1937 {
    __slate_anon_0: __SlateRecord_bpf_h_1938,
    __slate_anon_1: __SlateRecord_bpf_h_1939,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1938 {
    __empty_insns: __SlateRecord_bpf_h_1938_2,
    insns: [sock_filter; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1938_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1939 {
    __empty_insnsi: __SlateRecord_bpf_h_1939_2,
    insnsi: [bpf_insn; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_bpf_h_1939_2 {
}


#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ma_write: [u8; 9] = [109, 97, 95, 119, 114, 105, 116, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ma_write_485: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ma_write) }) as *mut ();

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_1964 {
    rcu: callback_head,
    work: work_struct,
}


fn trace_ma_write(r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ma_write.key) }) != 0 {
__do_trace_ma_write(r#fn, mas, piv, val);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_bpf_h_2532 {
    cgroup_storage: [*mut bpf_cgroup_storage; 2],
    bpf_cookie: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_cgroup_storage {
}


#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_486: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_ma_op: [u8; 6] = [109, 97, 95, 111, 112, 0];

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_ma_op: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_ma_op) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ma_op) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>>(__SCT__tp_func_ma_op as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>, *mut ()>(Some(__traceiter_ma_op)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state)>, *mut ()>(Some(__probestub_ma_op)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_76 {
    next: *mut perf_raw_frag,
    pad: u64,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.__probestub_ma_op")]
extern "C" fn __probestub_ma_op(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_ma_op: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>, *mut ()>(Some(__traceiter_ma_op)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_ma_read: [u8; 8] = [109, 97, 95, 114, 101, 97, 100, 0];

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_ma_read: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_ma_read) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ma_read) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>>(__SCT__tp_func_ma_read as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>, *mut ()>(Some(__traceiter_ma_read)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state)>, *mut ()>(Some(__probestub_ma_read)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_149 {
    __slate_anon_0: __SlateRecord_perf_event_h_150,
    __slate_anon_1: __SlateRecord_perf_event_h_165,
    __slate_anon_2: __SlateRecord_perf_event_h_174,
    __slate_anon_3: __SlateRecord_perf_event_h_177,
    __slate_anon_4: __SlateRecord_perf_event_h_181,
    __slate_anon_5: __SlateRecord_perf_event_h_186,
    __slate_anon_6: __SlateRecord_perf_event_h_196,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_150 {
    config: u64,
    config1: u64,
    last_tag: u64,
    dyn_constraint: u64,
    config_base: u64,
    event_base: u64,
    event_base_rdpmc: i32,
    idx: i32,
    last_cpu: i32,
    flags: i32,
    extra_reg: hw_perf_event_extra,
    branch_reg: hw_perf_event_extra,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_165 {
    aux_config: u64,
    aux_paused: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_174 {
    hrtimer: hrtimer,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_177 {
    tp_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_181 {
    pwr_acc: u64,
    ptsc: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_186 {
    info: arch_hw_breakpoint,
    bp_list: rhlist_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_196 {
    iommu_bank: u8,
    iommu_cntr: u8,
    padding: u16,
    conf: u64,
    conf1: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_244 {
    __slate_anon_0: __SlateRecord_perf_event_h_245,
    __slate_anon_1: __SlateRecord_perf_event_h_259,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_245 {
    last_period: u64,
    period_left: __SlateRecord_local64_h_26,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_259 {
    saved_metric: u64,
    saved_slots: u64,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.__probestub_ma_read")]
extern "C" fn __probestub_ma_read(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_ma_read: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state) -> i32>, *mut ()>(Some(__traceiter_ma_read)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

#[unsafe(link_section = "__tracepoints_strings")]
static __tpstrtab_ma_write: [u8; 9] = [109, 97, 95, 119, 114, 105, 116, 101, 0];

#[unsafe(no_mangle)]
#[used(compiler)]
#[unsafe(link_section = "__tracepoints")]
static mut __tracepoint_ma_write: tracepoint = tracepoint { name: unsafe { core::ptr::addr_of!(__tpstrtab_ma_write) as *const u8 }, key: static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} }, static_call_key: unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ma_write) }, static_call_tramp: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ()) -> i32>, *mut ()>(unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ()) -> i32>>(__SCT__tp_func_ma_write as *const ()) }) }, iterator: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ()) -> i32>, *mut ()>(Some(__traceiter_ma_write)) }, probestub: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ())>, *mut ()>(Some(__probestub_ma_write)) }, funcs: core::ptr::null_mut::<tracepoint_func>(), ext: core::ptr::null_mut::<tracepoint_ext>() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.__probestub_ma_write")]
extern "C" fn __probestub_ma_write(__data: *mut (), r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) {
}

#[unsafe(no_mangle)]
static mut __SCK__tp_func_ma_write: static_call_key = static_call_key { func: unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ()) -> i32>, *mut ()>(Some(__traceiter_ma_write)) }, __slate_anon_1: __SlateRecord_static_call_types_h_65 { r#type: 1u64 } };

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
struct perf_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct event_filter {
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
union __SlateRecord_perf_event_h_1156 {
    flags: u64,
    aux_flags: u64,
    __slate_anon_2: __SlateRecord_perf_event_h_1159,
}


#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1159 {
    __slate_bits_0: __slate_bits::__SlateBits1051U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_perf_event_h_1163 {
    addr: *mut (),
    head: u64,
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
struct __SlateRecord_perf_event_h_1315 {
    pid: u32,
    tid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_perf_event_h_1321 {
    cpu: u32,
    reserved: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_array {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct array_buffer {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracer {
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
struct __SlateRecord_local_h_12 {
    a: __SlateRecord_types_h_197,
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
union __SlateRecord_trace_events_h_275 {
    __slate_anon_0: __SlateRecord_trace_events_h_276,
    define_fields: Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_trace_events_h_276 {
    name: *const u8,
    size: i32,
    align: i32,
    __slate_bits_0: __slate_bits::__SlateBits1069U0,
    filter_type: i32,
    len: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ring_buffer_event {
    __slate_bits_0: __slate_bits::__SlateBits782U0,
    array: [u32; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_buf {
    buffer: *mut u8,
    size: u64,
    len: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_trace_events_h_379 {
    name: *const u8,
    tp: *mut tracepoint,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_trace_events_h_390 {
    module: *mut (),
    refcnt: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_subsystem_dir {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct eventfs_inode {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_seq {
    seq: seq_buf,
    readpos: u64,
    full: i32,
    buffer: [u8; 8156],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_attr {
    r#type: u32,
    size: u32,
    config: u64,
    __slate_anon_3: __SlateRecord_perf_event_h_410,
    sample_type: u64,
    read_format: u64,
    __slate_bits_0: __slate_bits::__SlateBits784U0,
    __slate_anon_46: __SlateRecord_perf_event_h_471,
    bp_type: u32,
    __slate_anon_48: __SlateRecord_perf_event_h_477,
    __slate_anon_49: __SlateRecord_perf_event_h_483,
    branch_sample_type: u64,
    sample_regs_user: u64,
    sample_stack_user: u32,
    clockid: i32,
    sample_regs_intr: u64,
    aux_watermark: u32,
    sample_max_stack: u16,
    __reserved_2: u16,
    aux_sample_size: u32,
    __slate_anon_59: __SlateRecord_perf_event_h_530,
    sig_data: u64,
    config3: u64,
    config4: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union perf_mem_data_src {
    val: u64,
    __slate_anon_1: __SlateRecord_perf_event_h_1322,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_branch_entry {
    from: u64,
    to: u64,
    __slate_bits_0: __slate_bits::__SlateBits793U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union perf_sample_weight {
    full: u64,
    __slate_anon_1: __SlateRecord_perf_event_h_1512,
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
struct __SlateRecord_maple_tree_h_20 {
    b: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_21 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_22 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_23 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_24 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_25 {
    b: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_53 {
    b: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_54 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_55 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_56 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_57 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_58 {
    b: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_88 {
    b: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_89 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_90 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_91 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_92 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_93 {
    b: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_94 {
    b: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_maple_tree_h_95 {
    b: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_fn__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_min__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_max__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_index__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_last__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_node__1 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_fn__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_min__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_max__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_index__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_last__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_node__1_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_fn__1_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_min__1_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_max__1_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_index__1_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_last__1_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_piv {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_val {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct _test_no_array_node__1_3 {
}


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits641U0 {
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
    pub struct __SlateBits479U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits293U0 {
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
    pub struct __SlateBits70U0 {
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
    pub struct __SlateBits70U1 {
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
    pub struct __SlateBits446U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits501U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits827U0 {
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
    pub struct __SlateBits296U0 {
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
    pub struct __SlateBits70U2 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits528U0 {
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
    pub struct __SlateBits525U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits528U1 {
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
    pub struct __SlateBits528U2 {
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
    pub struct __SlateBits528U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1037U0 {
        #[bits(1)]
        pub embedded: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1051U0 {
        #[bits(1)]
        pub skip_read: u64,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits784U0 {
        #[bits(1)]
        pub disabled: u64,
        #[bits(1)]
        pub inherit: u64,
        #[bits(1)]
        pub pinned: u64,
        #[bits(1)]
        pub exclusive: u64,
        #[bits(1)]
        pub exclude_user: u64,
        #[bits(1)]
        pub exclude_kernel: u64,
        #[bits(1)]
        pub exclude_hv: u64,
        #[bits(1)]
        pub exclude_idle: u64,
        #[bits(1)]
        pub mmap: u64,
        #[bits(1)]
        pub comm: u64,
        #[bits(1)]
        pub freq: u64,
        #[bits(1)]
        pub inherit_stat: u64,
        #[bits(1)]
        pub enable_on_exec: u64,
        #[bits(1)]
        pub task: u64,
        #[bits(1)]
        pub watermark: u64,
        #[bits(2)]
        pub precise_ip: u64,
        #[bits(1)]
        pub mmap_data: u64,
        #[bits(1)]
        pub sample_id_all: u64,
        #[bits(1)]
        pub exclude_host: u64,
        #[bits(1)]
        pub exclude_guest: u64,
        #[bits(1)]
        pub exclude_callchain_kernel: u64,
        #[bits(1)]
        pub exclude_callchain_user: u64,
        #[bits(1)]
        pub mmap2: u64,
        #[bits(1)]
        pub comm_exec: u64,
        #[bits(1)]
        pub use_clockid: u64,
        #[bits(1)]
        pub context_switch: u64,
        #[bits(1)]
        pub write_backward: u64,
        #[bits(1)]
        pub namespaces: u64,
        #[bits(1)]
        pub ksymbol: u64,
        #[bits(1)]
        pub bpf_event: u64,
        #[bits(1)]
        pub aux_output: u64,
        #[bits(1)]
        pub cgroup: u64,
        #[bits(1)]
        pub text_poke: u64,
        #[bits(1)]
        pub build_id: u64,
        #[bits(1)]
        pub inherit_thread: u64,
        #[bits(1)]
        pub remove_on_exec: u64,
        #[bits(1)]
        pub sigtrap: u64,
        #[bits(1)]
        pub defer_callchain: u64,
        #[bits(1)]
        pub defer_output: u64,
        #[bits(24)]
        pub __reserved_1: u64,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits790U0 {
        #[bits(1)]
        pub aux_start_paused: u32,
        #[bits(1)]
        pub aux_pause: u32,
        #[bits(1)]
        pub aux_resume: u32,
        #[bits(29)]
        pub __reserved_3: u32,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits793U0 {
        #[bits(1)]
        pub mispred: u64,
        #[bits(1)]
        pub predicted: u64,
        #[bits(1)]
        pub in_tx: u64,
        #[bits(1)]
        pub abort: u64,
        #[bits(16)]
        pub cycles: u64,
        #[bits(4)]
        pub r#type: u64,
        #[bits(2)]
        pub spec: u64,
        #[bits(4)]
        pub new_type: u64,
        #[bits(3)]
        pub r#priv: u64,
        #[bits(31)]
        pub reserved: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits792U0 {
        #[bits(5)]
        pub mem_op: u64,
        #[bits(14)]
        pub mem_lvl: u64,
        #[bits(5)]
        pub mem_snoop: u64,
        #[bits(2)]
        pub mem_lock: u64,
        #[bits(7)]
        pub mem_dtlb: u64,
        #[bits(4)]
        pub mem_lvl_num: u64,
        #[bits(1)]
        pub mem_remote: u64,
        #[bits(2)]
        pub mem_snoopx: u64,
        #[bits(3)]
        pub mem_blk: u64,
        #[bits(3)]
        pub mem_hops: u64,
        #[bits(5)]
        pub mem_region: u64,
        #[bits(13)]
        pub mem_rsvd: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits135U0 {
        #[bits(16)]
        pub cs: u64,
        #[bits(2)]
        pub sl: u64,
        #[bits(1)]
        pub wfe: u64,
        #[bits(45, access = na)]
        pub __slate_pad_3: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits136U0 {
        #[bits(16)]
        pub ss: u64,
        #[bits(1)]
        pub sti: u64,
        #[bits(1)]
        pub swevent: u64,
        #[bits(1)]
        pub nmi: u64,
        #[bits(13, access = na)]
        pub __slate_pad_4: u16,
        #[bits(8)]
        pub vector: u64,
        #[bits(8, access = na)]
        pub __slate_pad_6: u8,
        #[bits(4)]
        pub r#type: u64,
        #[bits(4, access = na)]
        pub __slate_pad_8: u8,
        #[bits(1)]
        pub enclave: u64,
        #[bits(1)]
        pub l: u64,
        #[bits(1)]
        pub nested: u64,
        #[bits(1, access = na)]
        pub __slate_pad_12: u8,
        #[bits(4)]
        pub insnlen: u64,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits930U0 {
        #[bits(1)]
        pub jited: u32,
        #[bits(1)]
        pub jit_requested: u32,
        #[bits(1)]
        pub jit_required: u32,
        #[bits(1)]
        pub gpl_compatible: u32,
        #[bits(1)]
        pub cb_access: u32,
        #[bits(1)]
        pub dst_needed: u32,
        #[bits(1)]
        pub blinding_requested: u32,
        #[bits(1)]
        pub blinded: u32,
        #[bits(1)]
        pub is_func: u32,
        #[bits(1)]
        pub kprobe_override: u32,
        #[bits(1)]
        pub has_callchain_buf: u32,
        #[bits(1)]
        pub enforce_expected_attach_type: u32,
        #[bits(1)]
        pub call_get_stack: u32,
        #[bits(1)]
        pub call_get_func_ip: u32,
        #[bits(1)]
        pub call_session_cookie: u32,
        #[bits(1)]
        pub tstamp_type_access: u32,
        #[bits(1)]
        pub sleepable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_17: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits842U0 {
        #[bits(4)]
        pub dst_reg: u8,
        #[bits(4)]
        pub src_reg: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits862U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits865U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits898U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits899U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits913U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits914U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits918U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits919U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits1069U0 {
        #[bits(1)]
        pub is_signed: u32,
        #[bits(1)]
        pub needs_test: u32,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits228U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits123U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits503U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits782U0 {
        #[bits(5)]
        pub type_len: u32,
        #[bits(27)]
        pub time_delta: u32,
    }
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
struct __SlateRecord_local64_h_26 {
    a: __SlateRecord_local_h_12,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct arch_hw_breakpoint {
    address: u64,
    mask: u64,
    len: u8,
    r#type: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_regs {
    abi: u64,
    regs: *mut pt_regs,
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
    __slate_bits_0: __slate_bits::__SlateBits827U0,
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


// double word (64-bit)
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_insn {
    code: u8,
    __slate_bits_0: __slate_bits::__SlateBits842U0,
    off: i16,
    imm: i32,
}


// 1 << 13
// insn[0].src_reg:  BPF_PSEUDO_FUNC
// insn[0].imm:      insn offset to the func
// insn[1].imm:      0
// insn[0].off:      0
// insn[1].off:      0
// ldimm64 rewrite:  address of the function
// verifier type:    PTR_TO_FUNC.
#[repr(C)]
#[derive(Clone, Copy)]
union bpf_attr {
    __slate_anon_0: __SlateRecord_bpf_h_1528,
    __slate_anon_1: __SlateRecord_bpf_h_1581,
    batch: __SlateRecord_bpf_h_1591,
    __slate_anon_3: __SlateRecord_bpf_h_1608,
    __slate_anon_4: __SlateRecord_bpf_h_1674,
    __slate_anon_5: __SlateRecord_bpf_h_1687,
    test: __SlateRecord_bpf_h_1703,
    __slate_anon_7: __SlateRecord_bpf_h_1727,
    info: __SlateRecord_bpf_h_1740,
    query: __SlateRecord_bpf_h_1746,
    raw_tracepoint: __SlateRecord_bpf_h_1769,
    __slate_anon_11: __SlateRecord_bpf_h_1776,
    task_fd_query: __SlateRecord_bpf_h_1794,
    link_create: __SlateRecord_bpf_h_1810,
    link_update: __SlateRecord_bpf_h_1895,
    link_detach: __SlateRecord_bpf_h_1916,
    enable_stats: __SlateRecord_bpf_h_1920,
    iter_create: __SlateRecord_bpf_h_1924,
    prog_bind_map: __SlateRecord_bpf_h_1929,
    token_create: __SlateRecord_bpf_h_1935,
    prog_stream_read: __SlateRecord_bpf_h_1940,
    prog_assoc_struct_ops: __SlateRecord_bpf_h_1947,
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link_info {
    r#type: u32,
    id: u32,
    prog_id: u32,
    __slate_anon_3: __SlateRecord_bpf_h_6842,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_info {
    insn_off: u32,
    type_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_line_info {
    insn_off: u32,
    file_name_off: u32,
    line_off: u32,
    line_col: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock_filter {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_type {
    name_off: u32,
    info: u32,
    __slate_anon_2: __SlateRecord_btf_h_64,
}


// BTF_KIND_INT is followed by a u32 and the following
// is the 32 bits arrangement:
// Attributes stored in the BTF_INT_ENCODING
#[repr(C)]
#[derive(Clone, Copy)]
struct btf_member {
    name_off: u32,
    r#type: u32,
    offset: u32,
}


// If the struct/union type info kind_flag is set, the
// following two macros are used to access bitfield_size
// and bit_offset from btf_member.offset.
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
    __slate_bits_0: __slate_bits::__SlateBits641U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_iter_seq_info {
    seq_ops: *const seq_operations,
    init_seq_private: Option<unsafe extern "C" fn(*mut (), *mut bpf_iter_aux_info) -> i32>,
    fini_seq_private: Option<unsafe extern "C" fn(*mut ())>,
    seq_priv_size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_ops {
    map_alloc_check: Option<unsafe extern "C" fn(*mut bpf_attr) -> i32>,
    map_alloc: Option<unsafe extern "C" fn(*mut bpf_attr) -> *mut bpf_map>,
    map_release: Option<unsafe extern "C" fn(*mut bpf_map, *mut file)>,
    map_free: Option<unsafe extern "C" fn(*mut bpf_map)>,
    map_get_next_key: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut ()) -> i32>,
    map_release_uref: Option<unsafe extern "C" fn(*mut bpf_map)>,
    map_lookup_elem_sys_only: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> *mut ()>,
    map_lookup_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_lookup_and_delete_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut (), u64) -> i32>,
    map_lookup_and_delete_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_update_batch: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_delete_batch: Option<unsafe extern "C" fn(*mut bpf_map, *const bpf_attr, *mut bpf_attr) -> i32>,
    map_lookup_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> *mut ()>,
    map_update_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut (), u64) -> i64>,
    map_delete_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_push_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), u64) -> i64>,
    map_pop_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_peek_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut ()) -> i64>,
    map_lookup_percpu_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), u32) -> *mut ()>,
    map_get_hash: Option<unsafe extern "C" fn(*mut bpf_map) -> i32>,
    map_fd_get_ptr: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, i32) -> *mut ()>,
    map_fd_put_ptr: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), bool)>,
    map_gen_lookup: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_insn) -> i32>,
    map_fd_sys_lookup_elem: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    map_seq_show_elem: Option<unsafe extern "C" fn(*mut bpf_map, *mut (), *mut seq_file)>,
    map_check_btf: Option<unsafe extern "C" fn(*mut bpf_map, *const btf, *const btf_type, *const btf_type) -> i32>,
    map_poke_track: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_prog_aux) -> i32>,
    map_poke_untrack: Option<unsafe extern "C" fn(*mut bpf_map, *mut bpf_prog_aux)>,
    map_poke_run: Option<unsafe extern "C" fn(*mut bpf_map, u32, *mut bpf_prog, *mut bpf_prog)>,
    map_direct_value_addr: Option<unsafe extern "C" fn(*const bpf_map, *mut u64, u32) -> i32>,
    map_direct_value_meta: Option<unsafe extern "C" fn(*const bpf_map, u64, *mut u32) -> i32>,
    map_mmap: Option<unsafe extern "C" fn(*mut bpf_map, *mut vm_area_struct) -> i32>,
    map_mmap_fault: Option<unsafe extern "C" fn(*mut bpf_map, *mut vm_fault) -> u32>,
    map_poll: Option<unsafe extern "C" fn(*mut bpf_map, *mut file, *mut poll_table_struct) -> u32>,
    map_get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
    map_local_storage_charge: Option<unsafe extern "C" fn(*mut bpf_local_storage_map, *mut (), u32) -> i32>,
    map_local_storage_uncharge: Option<unsafe extern "C" fn(*mut bpf_local_storage_map, *mut (), u32)>,
    map_owner_storage_ptr: Option<unsafe extern "C" fn(*mut ()) -> *mut *mut bpf_local_storage>,
    map_redirect: Option<unsafe extern "C" fn(*mut bpf_map, u64, u64) -> i64>,
    map_meta_equal: Option<unsafe extern "C" fn(*const bpf_map, *const bpf_map) -> bool>,
    map_set_for_each_callback_args: Option<unsafe extern "C" fn(*mut bpf_verifier_env, *mut bpf_func_state, *mut bpf_func_state) -> i32>,
    map_for_each_callback: Option<unsafe extern "C" fn(*mut bpf_map, Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>, *mut (), u64) -> i64>,
    map_mem_usage: Option<unsafe extern "C" fn(*const bpf_map) -> u64>,
    map_btf_id: *mut i32,
    iter_seq_info: *const bpf_iter_seq_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field_kptr {
    btf: *mut btf,
    module: *mut module,
    dtor: Option<unsafe extern "C" fn(*mut ())>,
    btf_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field_graph_root {
    btf: *mut btf,
    value_btf_id: u32,
    node_offset: u32,
    value_rec: *mut btf_record,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_field {
    offset: u32,
    size: u32,
    r#type: u32,
    __slate_anon_3: __SlateRecord_bpf_h_256,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_record {
    cnt: u32,
    field_mask: u32,
    spin_lock_off: i32,
    res_spin_lock_off: i32,
    timer_off: i32,
    wq_off: i32,
    refcount_off: i32,
    task_work_off: i32,
    fields: [btf_field; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map_owner {
    r#type: u32,
    jited: bool,
    xdp_has_frags: bool,
    sleepable: bool,
    storage_cookie: [u64; 2],
    attach_func_proto: *const btf_type,
    expected_attach_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_map {
    sha: [u8; 32],
    ops: *const bpf_map_ops,
    inner_map_meta: *mut bpf_map,
    security: *mut (),
    map_type: u32,
    key_size: u32,
    value_size: u32,
    max_entries: u32,
    map_extra: u64,
    map_flags: u32,
    id: u32,
    record: *mut btf_record,
    numa_node: i32,
    btf_key_type_id: u32,
    btf_value_type_id: u32,
    btf_vmlinux_value_type_id: u32,
    btf: *mut btf,
    name: [u8; 16],
    freeze_mutex: mutex,
    refcnt: __SlateRecord_types_h_197,
    usercnt: __SlateRecord_types_h_197,
    __slate_anon_21: __SlateRecord_bpf_h_329,
    writecnt: __SlateRecord_types_h_197,
    owner_lock: spinlock,
    owner: *mut bpf_map_owner,
    bypass_spec_v1: bool,
    frozen: bool,
    free_after_mult_rcu_gp: bool,
    free_after_rcu_gp: bool,
    sleepable_refcnt: __SlateRecord_types_h_197,
    elem_count: *mut i64,
    cookie: u64,
    excl_prog_sha: *mut u8,
}


// bpf_type_flag contains a set of flags that are applicable to the values of
// arg_type, ret_type and reg_type. For example, a pointer value may be null,
// or a memory is read-only. We classify types into two categories: base types
// and extended types. Extended types are base types combined with a type flag.
//
// Currently there are no more than 32 base types in arg_type, ret_type and
// reg_types.
// Max number of base types.
// Max number of all types.
// The longest tracepoint has 12 args.
// See include/trace/bpf_probe.h
//
// Also reuse this macro for maximum number of arguments a BPF function
// or a kfunc can have. Args 1-5 are passed in registers, args 6-12 via
// stack arg slots. The JIT may map some stack arg slots to registers based
// on the native calling convention (e.g., arg 6 to R9 on x86-64).
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_proto {
    func: Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>,
    gpl_only: bool,
    pkt_access: bool,
    might_sleep: bool,
    allow_fastcall: bool,
    ret_type: u32,
    __slate_anon_6: __SlateRecord_bpf_h_996,
    __slate_anon_7: __SlateRecord_bpf_h_1006,
    ret_btf_id: *mut i32,
    allowed: Option<unsafe extern "C" fn(*const bpf_prog) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_insn_access_aux {
    reg_type: u32,
    is_ldsx: bool,
    __slate_anon_2: __SlateRecord_bpf_h_1109,
    log: *mut bpf_verifier_log,
    is_retval: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_ops {
    test_run: Option<unsafe extern "C" fn(*mut bpf_prog, *const bpf_attr, *mut bpf_attr) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_verifier_ops {
    get_func_proto: Option<unsafe extern "C" fn(u32, *const bpf_prog) -> *const bpf_func_proto>,
    is_valid_access: Option<unsafe extern "C" fn(i32, i32, u32, *const bpf_prog, *mut bpf_insn_access_aux) -> bool>,
    gen_prologue: Option<unsafe extern "C" fn(*mut bpf_insn, bool, *const bpf_prog) -> i32>,
    gen_epilogue: Option<unsafe extern "C" fn(*mut bpf_insn, *const bpf_prog, i16) -> i32>,
    gen_ld_abs: Option<unsafe extern "C" fn(*const bpf_insn, *mut bpf_insn) -> i32>,
    convert_ctx_access: Option<unsafe extern "C" fn(u32, *const bpf_insn, *mut bpf_insn, *mut bpf_prog, *mut u32) -> u32>,
    btf_struct_access: Option<unsafe extern "C" fn(*mut bpf_verifier_log, *const bpf_reg_state, i32, i32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_offload {
    prog: *mut bpf_prog,
    netdev: *mut net_device,
    offdev: *mut bpf_offload_dev,
    dev_priv: *mut (),
    offloads: list_head,
    dev_state: bool,
    opt_failed: bool,
    jited_image: *mut (),
    jited_len: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_func_model {
    ret_size: u8,
    ret_flags: u8,
    nr_args: u8,
    arg_size: [u8; 12],
    arg_flags: [u8; 12],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_ksym {
    start: u64,
    end: u64,
    name: [u8; 512],
    lnode: list_head,
    tnode: latch_tree_node,
    prog: bool,
    fp_start: u32,
    fp_end: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_tramp_skip {
    prog: *mut bpf_prog,
    nop: *mut (),
    target: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_tramp_image {
    image: *mut (),
    size: i32,
    ksym: bpf_ksym,
    pcref: percpu_ref,
    call_orig: bool,
    tr: *mut bpf_trampoline,
    list: list_head,
    nr_skips: i32,
    skips: *mut bpf_tramp_skip,
    __slate_anon_9: __SlateRecord_bpf_h_1393,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_trampoline {
    hlist_key: hlist_node,
    hlist_ip: hlist_node,
    fops: *mut ftrace_ops,
    refcnt: refcount_struct,
    flags: u32,
    key: u64,
    ip: u64,
    func: __SlateRecord_bpf_h_1423,
    extension_prog: *mut bpf_prog,
    progs_hlist: [hlist_head; 3],
    progs_cnt: [i32; 3],
    cur_image: *mut bpf_tramp_image,
    images: list_head,
    multi_attach: __SlateRecord_bpf_h_1442,
}


// Fits in 2048B
#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_func_info_aux {
    linkage: u16,
    unreliable: bool,
    called: [bool; 2],
    verified: [bool; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_jit_poke_descriptor {
    tailcall_target: *mut (),
    tailcall_bypass: *mut (),
    bypass_addr: *mut (),
    aux: *mut (),
    __slate_anon_4: __SlateRecord_bpf_h_1705,
    tailcall_target_stable: bool,
    adj_off: u8,
    reason: u16,
    insn_idx: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_ctx_arg_aux {
    offset: u32,
    reg_type: u32,
    btf: *mut btf,
    btf_id: u32,
    ref_id: u32,
    refcounted: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct btf_mod_pair {
    btf: *mut btf,
    module: *mut module,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_stream {
    capacity: __SlateRecord_types_h_190,
    log: llist_head,
    lock: mutex,
    backlog_head: *mut llist_node,
    backlog_tail: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_aux {
    refcnt: __SlateRecord_types_h_197,
    used_map_cnt: u32,
    used_btf_cnt: u32,
    max_ctx_offset: u32,
    max_pkt_offset: u32,
    max_tp_access: u32,
    stack_depth: u32,
    id: u32,
    func_cnt: u32,
    real_func_cnt: u32,
    func_idx: u32,
    attach_btf_id: u32,
    attach_st_ops_member_off: u32,
    ctx_arg_info_size: u32,
    max_rdonly_access: u32,
    max_rdwr_access: u32,
    subprog_start: u32,
    attach_btf: *mut btf,
    ctx_arg_info: *mut bpf_ctx_arg_aux,
    priv_stack_ptr: *mut (),
    dst_mutex: mutex,
    dst_prog: *mut bpf_prog,
    dst_trampoline: *mut bpf_trampoline,
    saved_dst_prog_type: u32,
    saved_dst_attach_type: u32,
    verifier_zext: bool,
    dev_bound: bool,
    offload_requested: bool,
    attach_btf_trace: bool,
    attach_tracing_prog: bool,
    tramp_linked: bool,
    func_proto_unreliable: bool,
    tail_call_reachable: bool,
    xdp_has_frags: bool,
    exception_cb: bool,
    exception_boundary: bool,
    is_extended: bool,
    jits_use_priv_stack: bool,
    priv_stack_requested: bool,
    changes_pkt_data: bool,
    might_sleep: bool,
    kprobe_write_ctx: bool,
    sig: __SlateRecord_bpf_h_1821_2,
    prog_array_member_cnt: u64,
    ext_mutex: mutex,
    arena: *mut bpf_arena,
    recursion_detected: Option<unsafe extern "C" fn(*mut bpf_prog)>,
    attach_func_proto: *const btf_type,
    attach_func_name: *const u8,
    func: *mut *mut bpf_prog,
    main_prog_aux: *mut bpf_prog_aux,
    jit_data: *mut (),
    poke_tab: *mut bpf_jit_poke_descriptor,
    kfunc_tab: *mut bpf_kfunc_desc_tab,
    kfunc_btf_tab: *mut bpf_kfunc_btf_tab,
    size_poke_tab: u32,
    ksym: bpf_ksym,
    ops: *const bpf_prog_ops,
    st_ops: *const bpf_struct_ops,
    used_maps: *mut *mut bpf_map,
    used_maps_mutex: mutex,
    used_btfs: *mut btf_mod_pair,
    prog: *mut bpf_prog,
    user: *mut user_struct,
    load_time: u64,
    verified_insns: u32,
    cgroup_atype: i32,
    cgroup_storage: [*mut bpf_map; 2],
    name: [u8; 16],
    bpf_exception_cb: Option<unsafe extern "C" fn(u64, u64, u64, u64, u64) -> u64>,
    stack_arg_sp_adjust: u16,
    security: *mut (),
    token: *mut bpf_token,
    offload: *mut bpf_prog_offload,
    btf: *mut btf,
    func_info: *mut bpf_func_info,
    func_info_aux: *mut bpf_func_info_aux,
    linfo: *mut bpf_line_info,
    jited_linfo: *mut *mut (),
    func_info_cnt: u32,
    nr_linfo: u32,
    linfo_idx: u32,
    r#mod: *mut module,
    num_exentries: u32,
    extable: *mut exception_table_entry,
    __slate_anon_85: __SlateRecord_bpf_h_1892,
    stream: [bpf_stream; 2],
    st_ops_assoc_mutex: mutex,
    st_ops_assoc: *mut bpf_map,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog {
    pages: u16,
    __slate_bits_0: __slate_bits::__SlateBits930U0,
    r#type: u32,
    expected_attach_type: u32,
    len: u32,
    jited_len: u32,
    __slate_anon_22: __SlateRecord_bpf_h_1926,
    stats: *mut bpf_prog_stats,
    active: *mut u8,
    bpf_func: Option<unsafe extern "C" fn(*const (), *const bpf_insn) -> u32>,
    aux: *mut bpf_prog_aux,
    orig_prog: *mut sock_fprog_kern,
    __slate_anon_28: __SlateRecord_bpf_h_1937,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link {
    refcnt: __SlateRecord_types_h_197,
    id: u32,
    r#type: u32,
    ops: *const bpf_link_ops,
    prog: *mut bpf_prog,
    flags: u32,
    attach_type: u32,
    __slate_anon_7: __SlateRecord_bpf_h_1964,
    sleepable: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_link_ops {
    release: Option<unsafe extern "C" fn(*mut bpf_link)>,
    dealloc: Option<unsafe extern "C" fn(*mut bpf_link)>,
    dealloc_deferred: Option<unsafe extern "C" fn(*mut bpf_link)>,
    detach: Option<unsafe extern "C" fn(*mut bpf_link) -> i32>,
    update_prog: Option<unsafe extern "C" fn(*mut bpf_link, *mut bpf_prog, *mut bpf_prog) -> i32>,
    show_fdinfo: Option<unsafe extern "C" fn(*const bpf_link, *mut seq_file)>,
    fill_link_info: Option<unsafe extern "C" fn(*const bpf_link, *mut bpf_link_info) -> i32>,
    update_map: Option<unsafe extern "C" fn(*mut bpf_link, *mut bpf_map, *mut bpf_map) -> i32>,
    poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_token {
    work: work_struct,
    refcnt: __SlateRecord_types_h_197,
    userns: *mut user_namespace,
    allowed_cmds: u64,
    allowed_maps: u64,
    allowed_progs: u64,
    allowed_attachs: u64,
    security: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_struct_ops {
    verifier_ops: *const bpf_verifier_ops,
    init: Option<unsafe extern "C" fn(*mut btf) -> i32>,
    check_member: Option<unsafe extern "C" fn(*const btf_type, *const btf_member, *const bpf_prog) -> i32>,
    init_member: Option<unsafe extern "C" fn(*const btf_type, *const btf_member, *mut (), *const ()) -> i32>,
    reg: Option<unsafe extern "C" fn(*mut (), *mut bpf_link) -> i32>,
    unreg: Option<unsafe extern "C" fn(*mut (), *mut bpf_link)>,
    update: Option<unsafe extern "C" fn(*mut (), *mut (), *mut bpf_link) -> i32>,
    validate: Option<unsafe extern "C" fn(*mut ()) -> i32>,
    cfi_stubs: *mut (),
    owner: *mut module,
    name: *const u8,
    func_models: [btf_func_model; 64],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_array_item {
    prog: *mut bpf_prog,
    __slate_anon_1: __SlateRecord_bpf_h_2532,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bpf_prog_array {
    rcu: callback_head,
    items: [bpf_prog_array_item; 0],
}


// Per-cpu temp buffers used by printf-like helpers to store the bprintf binary
// arguments representation.
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_callchain_entry {
    nr: u64,
    ip: [u64; 0],
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct perf_raw_frag {
    __slate_anon_0: __SlateRecord_perf_event_h_76,
    copy: Option<unsafe extern "C" fn(*mut (), *const (), u64, u64) -> u64>,
    data: *mut (),
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_raw_record {
    frag: perf_raw_frag,
    size: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_branch_stack {
    nr: u64,
    hw_idx: u64,
    entries: [perf_branch_entry; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hw_perf_event_extra {
    config: u64,
    reg: u32,
    alloc: i32,
    idx: i32,
}


// hw_perf_event::flag values
//
// PERF_EVENT_FLAG_ARCH bits are reserved for architecture-specific
// usage.
#[repr(C)]
#[derive(Clone, Copy)]
struct hw_perf_event {
    __slate_anon_0: __SlateRecord_perf_event_h_149,
    target: *mut task_struct,
    addr_filters: *mut (),
    addr_filters_gen: u64,
    state: i32,
    prev_count: __SlateRecord_local64_h_26,
    sample_period: u64,
    __slate_anon_7: __SlateRecord_perf_event_h_244,
    interrupts_seq: u64,
    interrupts: u64,
    freq_time_stamp: u64,
    freq_count_stamp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pmu {
    entry: list_head,
    events_lock: spinlock,
    events: list_head,
    module: *mut module,
    dev: *mut device,
    parent: *mut device,
    attr_groups: *mut *const attribute_group,
    attr_update: *mut *const attribute_group,
    name: *const u8,
    r#type: i32,
    capabilities: i32,
    scope: u32,
    cpu_pmu_context: *mut *mut perf_cpu_pmu_context,
    exclusive_cnt: __SlateRecord_types_h_190,
    task_ctx_nr: i32,
    hrtimer_interval_ms: i32,
    nr_addr_filters: u32,
    pmu_enable: Option<unsafe extern "C" fn(*mut pmu)>,
    pmu_disable: Option<unsafe extern "C" fn(*mut pmu)>,
    event_init: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    event_mapped: Option<unsafe extern "C" fn(*mut perf_event, *mut mm_struct)>,
    event_unmapped: Option<unsafe extern "C" fn(*mut perf_event, *mut mm_struct)>,
    add: Option<unsafe extern "C" fn(*mut perf_event, i32) -> i32>,
    del: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    start: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    stop: Option<unsafe extern "C" fn(*mut perf_event, i32)>,
    read: Option<unsafe extern "C" fn(*mut perf_event)>,
    start_txn: Option<unsafe extern "C" fn(*mut pmu, u32)>,
    commit_txn: Option<unsafe extern "C" fn(*mut pmu) -> i32>,
    cancel_txn: Option<unsafe extern "C" fn(*mut pmu)>,
    event_idx: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    sched_task: Option<unsafe extern "C" fn(*mut perf_event_pmu_context, *mut task_struct, bool)>,
    task_ctx_cache: *mut kmem_cache,
    setup_aux: Option<unsafe extern "C" fn(*mut perf_event, *mut *mut (), i32, bool) -> *mut ()>,
    free_aux: Option<unsafe extern "C" fn(*mut ())>,
    snapshot_aux: Option<unsafe extern "C" fn(*mut perf_event, *mut perf_output_handle, u64) -> i64>,
    addr_filters_validate: Option<unsafe extern "C" fn(*mut list_head) -> i32>,
    addr_filters_sync: Option<unsafe extern "C" fn(*mut perf_event)>,
    aux_output_match: Option<unsafe extern "C" fn(*mut perf_event) -> i32>,
    filter: Option<unsafe extern "C" fn(*mut pmu, i32) -> bool>,
    check_period: Option<unsafe extern "C" fn(*mut perf_event, u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_addr_filters_head {
    list: list_head,
    lock: raw_spinlock,
    nr_file_filters: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_addr_filter_range {
    start: u64,
    size: u64,
}


// Event capabilities. For event_caps and groups caps.
//
// PERF_EV_CAP_SOFTWARE: Is a software event.
// PERF_EV_CAP_READ_ACTIVE_PKG: A CPU event (or cgroup event) that can be read
// from any CPU in the package where it is active.
// PERF_EV_CAP_SIBLING: An event with this flag must be a group sibling and
// cannot be a group leader. If an event with this flag is detached from the
// group it is scheduled out and moved into an unrecoverable ERROR state.
// PERF_EV_CAP_READ_SCOPE: A CPU event that can be read from any CPU of the
// PMU scope where it is active.
#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
    event_entry: list_head,
    sibling_list: list_head,
    active_list: list_head,
    group_node: rb_node,
    group_index: u64,
    migrate_entry: list_head,
    hlist_entry: hlist_node,
    active_entry: list_head,
    nr_siblings: i32,
    event_caps: i32,
    group_caps: i32,
    group_generation: u32,
    group_leader: *mut perf_event,
    pmu: *mut pmu,
    pmu_private: *mut (),
    state: i32,
    attach_state: u32,
    count: __SlateRecord_local64_h_26,
    child_count: __SlateRecord_types_h_197,
    total_time_enabled: u64,
    total_time_running: u64,
    tstamp: u64,
    attr: perf_event_attr,
    header_size: u16,
    id_header_size: u16,
    read_size: u16,
    hw: hw_perf_event,
    ctx: *mut perf_event_context,
    pmu_ctx: *mut perf_event_pmu_context,
    refcount: __SlateRecord_types_h_197,
    child_total_time_enabled: __SlateRecord_types_h_197,
    child_total_time_running: __SlateRecord_types_h_197,
    child_mutex: mutex,
    child_list: list_head,
    parent: *mut perf_event,
    oncpu: i32,
    cpu: i32,
    owner_entry: list_head,
    owner: *mut task_struct,
    mmap_mutex: mutex,
    mmap_count: refcount_struct,
    rb: *mut perf_buffer,
    rb_entry: list_head,
    rcu_batches: u64,
    rcu_pending: i32,
    waitq: wait_queue_head,
    fasync: *mut fasync_struct,
    pending_wakeup: u32,
    pending_kill: u32,
    pending_disable: u32,
    pending_addr: u64,
    pending_irq: irq_work,
    pending_disable_irq: irq_work,
    pending_task: callback_head,
    pending_work: u32,
    event_limit: __SlateRecord_types_h_190,
    addr_filters: perf_addr_filters_head,
    addr_filter_ranges: *mut perf_addr_filter_range,
    addr_filters_gen: u64,
    aux_event: *mut perf_event,
    destroy: Option<unsafe extern "C" fn(*mut perf_event)>,
    callback_head: callback_head,
    ns: *mut pid_namespace,
    id: u64,
    lost_samples: __SlateRecord_types_h_197,
    clock: Option<unsafe extern "C" fn() -> u64>,
    overflow_handler: Option<unsafe extern "C" fn(*mut perf_event, *mut perf_sample_data, *mut pt_regs)>,
    overflow_handler_context: *mut (),
    prog: *mut bpf_prog,
    bpf_cookie: u64,
    tp_event: *mut trace_event_call,
    filter: *mut event_filter,
    cgrp: *mut perf_cgroup,
    security: *mut (),
    sb_list: list_head,
    pmu_list: list_head,
    orig_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_pmu_context {
    pmu: *mut pmu,
    ctx: *mut perf_event_context,
    pmu_ctx_entry: list_head,
    pinned_active: list_head,
    flexible_active: list_head,
    __slate_bits_0: __slate_bits::__SlateBits1037U0,
    nr_events: u32,
    nr_cgroups: u32,
    nr_freq: u32,
    refcount: __SlateRecord_types_h_190,
    callback_head: callback_head,
    rotate_necessary: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_groups {
    tree: rb_root,
    index: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_time_ctx {
    time: u64,
    stamp: u64,
    offset: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event_context {
    lock: raw_spinlock,
    mutex: mutex,
    pmu_ctx_list: list_head,
    pinned_groups: perf_event_groups,
    flexible_groups: perf_event_groups,
    event_list: list_head,
    nr_events: i32,
    nr_user: i32,
    is_active: i32,
    nr_stat: i32,
    nr_freq: i32,
    rotate_disable: i32,
    refcount: refcount_struct,
    task: *mut task_struct,
    time: perf_time_ctx,
    timeguest: perf_time_ctx,
    parent_ctx: *mut perf_event_context,
    parent_gen: u64,
    generation: u64,
    pin_count: i32,
    nr_cgroups: i32,
    callback_head: callback_head,
    nr_no_switch_fast: __SlateRecord_local_h_12,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_ctx_data {
    callback_head: callback_head,
    refcount: refcount_struct,
    global: i32,
    ctx_cache: *mut kmem_cache,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cpu_pmu_context {
    epc: perf_event_pmu_context,
    task_epc: *mut perf_event_pmu_context,
    sched_cb_entry: list_head,
    sched_cb_usage: i32,
    active_oncpu: i32,
    exclusive: i32,
    pmu_disable_count: i32,
    hrtimer_lock: raw_spinlock,
    hrtimer: hrtimer,
    hrtimer_interval: i64,
    hrtimer_active: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_output_handle {
    event: *mut perf_event,
    rb: *mut perf_buffer,
    wakeup: u64,
    size: u64,
    __slate_anon_4: __SlateRecord_perf_event_h_1156,
    __slate_anon_5: __SlateRecord_perf_event_h_1163,
    page: i32,
}


core::arch::global_asm!("\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{0} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {1}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_ma_op\t\t\n__SCT__tp_func_ma_op:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {2} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_ma_op, @function\t\n.size __SCT__tp_func_ma_op, . - __SCT__tp_func_ma_op \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{3} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {4}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_ma_read\t\t\n__SCT__tp_func_ma_read:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {5} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_ma_read, @function\t\n.size __SCT__tp_func_ma_read, . - __SCT__tp_func_ma_read \n.popsection\t\t\t\t\t\n\n\t.section \"__tracepoints_ptrs\", \"a\"\t\t\n\t.balign 4\t\t\t\t\t\n\t.long \t{6} - .\t\t\n\t.previous\t\t\t\t\t\n\n.pushsection .discard.ibt_endbr_noseal\n\t .quad {7}\n\t.popsection\n\t\n.pushsection .static_call.text, \"ax\"\t\t\n.align 4\t\t\t\t\t\t\n.globl __SCT__tp_func_ma_write\t\t\n__SCT__tp_func_ma_write:\t\t\t\n911: .pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 911b - ., 1; .popsection\t\t\t\t\t\n.byte 0xe9; .long {8} - (. + 4)\t\t\t\t\t\t\n.byte 0x0f, 0xb9, 0xcc\t\t\t\t\n.type __SCT__tp_func_ma_write, @function\t\n.size __SCT__tp_func_ma_write, . - __SCT__tp_func_ma_write \n.popsection\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_alloc_cyclic: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_walk: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_empty_area: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_empty_area_rev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_store: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_store_gfp: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_store_prealloc: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_preallocate: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_destroy: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_next: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_next_range: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mt_next: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_prev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_prev_range: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mt_prev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_pause: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_find: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_find_range: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_find_rev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_find_range_rev: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mas_erase: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_load: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_store_range: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_store: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_insert_range: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_insert: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_alloc_range: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_alloc_cyclic: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_alloc_rrange: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_erase: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___mt_dup: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_dup: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol___mt_destroy: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mtree_destroy: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mt_find: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_mt_find_after: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous", sym __tracepoint_ma_op, sym __probestub_ma_op, sym __traceiter_ma_op, sym __tracepoint_ma_read, sym __probestub_ma_read, sym __traceiter_ma_read, sym __tracepoint_ma_write, sym __probestub_ma_write, sym __traceiter_ma_write, sym mas_alloc_cyclic, sym mas_walk, sym mas_empty_area, sym mas_empty_area_rev, sym mas_store, sym mas_store_gfp, sym mas_store_prealloc, sym mas_preallocate, sym mas_destroy, sym mas_next, sym mas_next_range, sym mt_next, sym mas_prev, sym mas_prev_range, sym mt_prev, sym mas_pause, sym mas_find, sym mas_find_range, sym mas_find_rev, sym mas_find_range_rev, sym mas_erase, sym mtree_load, sym mtree_store_range, sym mtree_store, sym mtree_insert_range, sym mtree_insert, sym mtree_alloc_range, sym mtree_alloc_cyclic, sym mtree_alloc_rrange, sym mtree_erase, sym __mt_dup, sym mtree_dup, sym __mt_destroy, sym mtree_destroy, sym mt_find, sym mt_find_after, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cgroup_info {
    time: perf_time_ctx,
    timeguest: perf_time_ctx,
    active: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_cgroup {
    css: cgroup_subsys_state,
    info: *mut perf_cgroup_info,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct perf_sample_data {
    sample_flags: u64,
    period: u64,
    dyn_size: u64,
    r#type: u64,
    tid_entry: __SlateRecord_perf_event_h_1315,
    time: u64,
    id: u64,
    cpu_entry: __SlateRecord_perf_event_h_1321,
    ip: u64,
    callchain: *mut perf_callchain_entry,
    raw: *mut perf_raw_record,
    br_stack: *mut perf_branch_stack,
    br_stack_cntr: *mut u64,
    weight: perf_sample_weight,
    data_src: perf_mem_data_src,
    txn: u64,
    regs_user: perf_regs,
    regs_intr: perf_regs,
    stack_user_size: u64,
    stream_id: u64,
    cgroup: u64,
    addr: u64,
    phys_addr: u64,
    data_page_size: u64,
    code_page_size: u64,
    aux_size: u64,
}


fn perf_fetch_caller_regs(regs: *mut pt_regs) {
perf_arch_fetch_caller_regs!(regs, (unsafe { __slate_intrinsic_ded315013ea6523(const { ((0 as i32) as u32) as i32 }) }) as u64);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct trace_entry {
    r#type: u16,
    flags: u8,
    preempt_count: u8,
    pid: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_iterator {
    tr: *mut trace_array,
    trace: *mut tracer,
    array_buffer: *mut array_buffer,
    private: *mut (),
    cpu_file: i32,
    mutex: mutex,
    buffer_iter: *mut *mut ring_buffer_iter,
    iter_flags: u64,
    temp: *mut (),
    temp_size: u32,
    fmt: *mut u8,
    fmt_size: u32,
    wait_index: __SlateRecord_types_h_190,
    tmp_seq: trace_seq,
    started: [cpumask; 1],
    closed: bool,
    snapshot: bool,
    seq: trace_seq,
    ent: *mut trace_entry,
    lost_events: u64,
    leftover: i32,
    ent_size: i32,
    cpu: i32,
    ts: u64,
    pos: i64,
    idx: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_functions {
    trace: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    raw: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    hex: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
    binary: Option<unsafe extern "C" fn(*mut trace_iterator, i32, *mut trace_event) -> u32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event {
    node: hlist_node,
    r#type: i32,
    funcs: *mut trace_event_functions,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_fields {
    r#type: *const u8,
    __slate_anon_1: __SlateRecord_trace_events_h_275,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_class {
    system: *const u8,
    probe: *mut (),
    perf_probe: *mut (),
    reg: Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>,
    fields_array: *mut trace_event_fields,
    get_fields: Option<unsafe extern "C" fn(*mut trace_event_call) -> *mut list_head>,
    fields: list_head,
    raw_init: Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_buffer {
    buffer: *mut trace_buffer,
    event: *mut ring_buffer_event,
    trace_file: *mut trace_event_file,
    entry: *mut (),
    trace_ctx: u32,
    regs: *mut pt_regs,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_call {
    list: list_head,
    class: *mut trace_event_class,
    __slate_anon_2: __SlateRecord_trace_events_h_379,
    event: trace_event,
    print_fmt: *mut u8,
    __slate_anon_5: __SlateRecord_trace_events_h_390,
    data: *mut (),
    flags: i32,
    perf_refcount: i32,
    perf_events: *mut hlist_head,
    prog_array: *mut bpf_prog_array,
    perf_perm: Option<unsafe extern "C" fn(*mut trace_event_call, *mut perf_event) -> i32>,
}


fn bpf_prog_array_valid(call: *mut trace_event_call) -> bool {
// This inline function checks whether call->prog_array
// is valid or not. The function is called in various places,
// outside rcu_read_lock/unlock, as a heuristic to speed up execution.
//
// If this function returns true, and later call->prog_array
// becomes false inside rcu_read_lock/unlock region,
// we bail out then. If this function return false,
// there is a risk that we might miss a few events if the checking
// were delayed until inside rcu_read_lock/unlock region and
// call->prog_array happened to become non-NULL then.
//
// Here, READ_ONCE() is used instead of rcu_access_pointer().
// rcu_access_pointer() requires the actual definition of
// "struct bpf_prog_array" while READ_ONCE() only needs
// a declaration of the same type.
let mut __v0: *mut bpf_prog_array;
loop {
if !(__native_word__24!() || ((const { core::mem::size_of::<*mut bpf_prog_array>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_683() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*call).prog_array) as *const *mut bpf_prog_array) };
!__v0.is_null()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_file {
    list: list_head,
    event_call: *mut trace_event_call,
    filter: *mut event_filter,
    ei: *mut eventfs_inode,
    tr: *mut trace_array,
    system: *mut trace_subsystem_dir,
    triggers: list_head,
    flags: u64,
    r#ref: refcount_struct,
    sm_ref: __SlateRecord_types_h_190,
    tm_ref: __SlateRecord_types_h_190,
}


#[inline(always)]
fn trace_trigger_soft_disabled(file: *mut trace_event_file) -> bool {
let eflags: u64 = unsafe { (*file).flags };
if likely!(eflags & 352u64 == 0) != 0 {
return false;
}
if likely__2!(eflags & 128) != 0 {
return false;
}
unsafe { __trace_trigger_soft_disabled(file) }
}

static str__maple_tree__trace_system_name: [u8; 11] = [109, 97, 112, 108, 101, 95, 116, 114, 101, 101, 0];

#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_ma_op {
    ent: trace_entry,
    r#fn: *const u8,
    min: u64,
    max: u64,
    index: u64,
    last: u64,
    node: *mut (),
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_ma_read {
    ent: trace_entry,
    r#fn: *const u8,
    min: u64,
    max: u64,
    index: u64,
    last: u64,
    node: *mut (),
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_raw_ma_write {
    ent: trace_entry,
    r#fn: *const u8,
    min: u64,
    max: u64,
    index: u64,
    last: u64,
    piv: u64,
    val: *mut (),
    node: *mut (),
    __data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_ma_op {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_ma_read {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_event_data_offsets_ma_write {
}


#[unsafe(link_section = ".text.slate_distinct.maple_tree.trace_raw_output_ma_op")]
extern "C" fn trace_raw_output_ma_op(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_ma_op = (unsafe { (*iter).ent }) as *mut trace_event_raw_ma_op;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"%s\tNode: %p (%lu %lu) range: %lu-%lu\n".as_ptr() as *const u8, (*field).r#fn, (*field).node, (*field).min, (*field).max, (*field).index, (*field).last) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_ma_op: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_ma_op);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.maple_tree.trace_raw_output_ma_read")]
extern "C" fn trace_raw_output_ma_read(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_ma_read = (unsafe { (*iter).ent }) as *mut trace_event_raw_ma_read;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"%s\tNode: %p (%lu %lu) range: %lu-%lu\n".as_ptr() as *const u8, (*field).r#fn, (*field).node, (*field).min, (*field).max, (*field).index, (*field).last) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_ma_read: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_ma_read);
__t0
};

#[unsafe(link_section = ".text.slate_distinct.maple_tree.trace_raw_output_ma_write")]
extern "C" fn trace_raw_output_ma_write(iter: *mut trace_iterator, flags: i32, trace_event: *mut trace_event) -> u32 {
let s: *mut trace_seq = unsafe { core::ptr::addr_of_mut!((*iter).seq) };
let mut ret: i32 = 0;
let field: *mut trace_event_raw_ma_write = (unsafe { (*iter).ent }) as *mut trace_event_raw_ma_write;
ret = unsafe { trace_raw_output_prep(iter, trace_event) };
if ret != 1 {
return ret as u32;
}
unsafe { trace_event_printf(iter, c"%s\tNode %p (%lu %lu) range:%lu-%lu piv (%lu) val %p\n".as_ptr() as *const u8, (*field).r#fn, (*field).node, (*field).min, (*field).max, (*field).index, (*field).last, (*field).piv, (*field).val) };
unsafe { trace_handle_return(s) }
}

static mut trace_event_type_funcs_ma_write: trace_event_functions = {
let mut __t0: trace_event_functions = unsafe { core::mem::zeroed() };
__t0.trace = Some(trace_raw_output_ma_write);
__t0
};

static mut trace_event_fields_ma_op: __SlateAlign16<[trace_event_fields; 7]> = __SlateAlign16([trace_event_fields { r#type: (b"const char *\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"fn\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<*const u8>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_20>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed(false as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t2: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t2.__slate_anon_0 = {
let mut __t3: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t3.name = (b"min\0".as_ptr() as *mut u8) as *const u8;
__t3.size = (core::mem::size_of::<u64>() as u64) as i32;
__t3.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_21>() as u64) as i32;
__t3.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t3.filter_type = 0i32;
__t3
};
__t2
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t4: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t4.__slate_anon_0 = {
let mut __t5: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t5.name = (b"max\0".as_ptr() as *mut u8) as *const u8;
__t5.size = (core::mem::size_of::<u64>() as u64) as i32;
__t5.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_22>() as u64) as i32;
__t5.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t5.filter_type = 0i32;
__t5
};
__t4
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t6: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t6.__slate_anon_0 = {
let mut __t7: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t7.name = (b"index\0".as_ptr() as *mut u8) as *const u8;
__t7.size = (core::mem::size_of::<u64>() as u64) as i32;
__t7.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_23>() as u64) as i32;
__t7.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t7.filter_type = 0i32;
__t7
};
__t6
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t8: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t8.__slate_anon_0 = {
let mut __t9: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t9.name = (b"last\0".as_ptr() as *mut u8) as *const u8;
__t9.size = (core::mem::size_of::<u64>() as u64) as i32;
__t9.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_24>() as u64) as i32;
__t9.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t9.filter_type = 0i32;
__t9
};
__t8
} }, trace_event_fields { r#type: (b"void *\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t10: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t10.__slate_anon_0 = {
let mut __t11: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t11.name = (b"node\0".as_ptr() as *mut u8) as *const u8;
__t11.size = (core::mem::size_of::<*mut ()>() as u64) as i32;
__t11.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_25>() as u64) as i32;
__t11.__slate_bits_0.__set_is_signed(false as u32);
__t11.filter_type = 0i32;
__t11
};
__t10
} }, {
let mut __t12: trace_event_fields = unsafe { core::mem::zeroed() };
__t12
}]);

static mut trace_event_fields_ma_read: __SlateAlign16<[trace_event_fields; 7]> = __SlateAlign16([trace_event_fields { r#type: (b"const char *\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"fn\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<*const u8>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_53>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed(false as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t2: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t2.__slate_anon_0 = {
let mut __t3: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t3.name = (b"min\0".as_ptr() as *mut u8) as *const u8;
__t3.size = (core::mem::size_of::<u64>() as u64) as i32;
__t3.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_54>() as u64) as i32;
__t3.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t3.filter_type = 0i32;
__t3
};
__t2
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t4: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t4.__slate_anon_0 = {
let mut __t5: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t5.name = (b"max\0".as_ptr() as *mut u8) as *const u8;
__t5.size = (core::mem::size_of::<u64>() as u64) as i32;
__t5.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_55>() as u64) as i32;
__t5.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t5.filter_type = 0i32;
__t5
};
__t4
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t6: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t6.__slate_anon_0 = {
let mut __t7: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t7.name = (b"index\0".as_ptr() as *mut u8) as *const u8;
__t7.size = (core::mem::size_of::<u64>() as u64) as i32;
__t7.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_56>() as u64) as i32;
__t7.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t7.filter_type = 0i32;
__t7
};
__t6
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t8: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t8.__slate_anon_0 = {
let mut __t9: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t9.name = (b"last\0".as_ptr() as *mut u8) as *const u8;
__t9.size = (core::mem::size_of::<u64>() as u64) as i32;
__t9.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_57>() as u64) as i32;
__t9.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t9.filter_type = 0i32;
__t9
};
__t8
} }, trace_event_fields { r#type: (b"void *\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t10: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t10.__slate_anon_0 = {
let mut __t11: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t11.name = (b"node\0".as_ptr() as *mut u8) as *const u8;
__t11.size = (core::mem::size_of::<*mut ()>() as u64) as i32;
__t11.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_58>() as u64) as i32;
__t11.__slate_bits_0.__set_is_signed(false as u32);
__t11.filter_type = 0i32;
__t11
};
__t10
} }, {
let mut __t12: trace_event_fields = unsafe { core::mem::zeroed() };
__t12
}]);

static mut trace_event_fields_ma_write: __SlateAlign16<[trace_event_fields; 9]> = __SlateAlign16([trace_event_fields { r#type: (b"const char *\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t0: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t1.name = (b"fn\0".as_ptr() as *mut u8) as *const u8;
__t1.size = (core::mem::size_of::<*const u8>() as u64) as i32;
__t1.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_88>() as u64) as i32;
__t1.__slate_bits_0.__set_is_signed(false as u32);
__t1.filter_type = 0i32;
__t1
};
__t0
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t2: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t2.__slate_anon_0 = {
let mut __t3: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t3.name = (b"min\0".as_ptr() as *mut u8) as *const u8;
__t3.size = (core::mem::size_of::<u64>() as u64) as i32;
__t3.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_89>() as u64) as i32;
__t3.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t3.filter_type = 0i32;
__t3
};
__t2
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t4: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t4.__slate_anon_0 = {
let mut __t5: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t5.name = (b"max\0".as_ptr() as *mut u8) as *const u8;
__t5.size = (core::mem::size_of::<u64>() as u64) as i32;
__t5.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_90>() as u64) as i32;
__t5.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t5.filter_type = 0i32;
__t5
};
__t4
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t6: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t6.__slate_anon_0 = {
let mut __t7: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t7.name = (b"index\0".as_ptr() as *mut u8) as *const u8;
__t7.size = (core::mem::size_of::<u64>() as u64) as i32;
__t7.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_91>() as u64) as i32;
__t7.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t7.filter_type = 0i32;
__t7
};
__t6
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t8: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t8.__slate_anon_0 = {
let mut __t9: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t9.name = (b"last\0".as_ptr() as *mut u8) as *const u8;
__t9.size = (core::mem::size_of::<u64>() as u64) as i32;
__t9.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_92>() as u64) as i32;
__t9.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t9.filter_type = 0i32;
__t9
};
__t8
} }, trace_event_fields { r#type: (b"unsigned long\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t10: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t10.__slate_anon_0 = {
let mut __t11: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t11.name = (b"piv\0".as_ptr() as *mut u8) as *const u8;
__t11.size = (core::mem::size_of::<u64>() as u64) as i32;
__t11.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_93>() as u64) as i32;
__t11.__slate_bits_0.__set_is_signed(((1i32.wrapping_neg() as u64) < 1u64) as u32);
__t11.filter_type = 0i32;
__t11
};
__t10
} }, trace_event_fields { r#type: (b"void *\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t12: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t12.__slate_anon_0 = {
let mut __t13: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t13.name = (b"val\0".as_ptr() as *mut u8) as *const u8;
__t13.size = (core::mem::size_of::<*mut ()>() as u64) as i32;
__t13.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_94>() as u64) as i32;
__t13.__slate_bits_0.__set_is_signed(false as u32);
__t13.filter_type = 0i32;
__t13
};
__t12
} }, trace_event_fields { r#type: (b"void *\0".as_ptr() as *mut u8) as *const u8, __slate_anon_1: {
let mut __t14: __SlateRecord_trace_events_h_275 = unsafe { core::mem::zeroed() };
__t14.__slate_anon_0 = {
let mut __t15: __SlateRecord_trace_events_h_276 = unsafe { core::mem::zeroed() };
__t15.name = (b"node\0".as_ptr() as *mut u8) as *const u8;
__t15.size = (core::mem::size_of::<*mut ()>() as u64) as i32;
__t15.align = (core::mem::align_of::<__SlateRecord_maple_tree_h_95>() as u64) as i32;
__t15.__slate_bits_0.__set_is_signed(false as u32);
__t15.filter_type = 0i32;
__t15
};
__t14
} }, {
let mut __t16: trace_event_fields = unsafe { core::mem::zeroed() };
__t16
}]);

fn trace_event_get_offsets_ma_op(__data_offsets: &trace_event_data_offsets_ma_op, r#fn: *const u8, mas: *mut ma_state) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn trace_event_get_offsets_ma_read(__data_offsets: &trace_event_data_offsets_ma_read, r#fn: *const u8, mas: *mut ma_state) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn trace_event_get_offsets_ma_write(__data_offsets: &trace_event_data_offsets_ma_write, r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) -> i32 {
let __data_size: i32 = 0;
__data_size
}

fn do_trace_event_raw_event_ma_op(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_ma_op = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_ma_op(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, r#fn, mas);
let entry: *mut trace_event_raw_ma_op = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 56u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_ma_op;
if entry.is_null() {
return;
}
unsafe {
(*entry).r#fn = r#fn;
}
unsafe {
(*entry).min = (*mas).min;
}
unsafe {
(*entry).max = (*mas).max;
}
unsafe {
(*entry).index = (*mas).index;
}
unsafe {
(*entry).last = (*mas).last;
}
unsafe {
(*entry).node = (*mas).node as *mut ();
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.maple_tree.trace_event_raw_event_ma_op")]
extern "C" fn trace_event_raw_event_ma_op(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
let mut __UNIQUE_ID_guard_684: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_684) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_ma_op(__data, r#fn, mas);
}

fn do_trace_event_raw_event_ma_read(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_ma_read = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_ma_read(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, r#fn, mas);
let entry: *mut trace_event_raw_ma_read = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 56u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_ma_read;
if entry.is_null() {
return;
}
unsafe {
(*entry).r#fn = r#fn;
}
unsafe {
(*entry).min = (*mas).min;
}
unsafe {
(*entry).max = (*mas).max;
}
unsafe {
(*entry).index = (*mas).index;
}
unsafe {
(*entry).last = (*mas).last;
}
unsafe {
(*entry).node = (*mas).node as *mut ();
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.maple_tree.trace_event_raw_event_ma_read")]
extern "C" fn trace_event_raw_event_ma_read(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
let mut __UNIQUE_ID_guard_685: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_685) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_ma_read(__data, r#fn, mas);
}

fn do_trace_event_raw_event_ma_write(__data: *mut (), r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) {
let trace_file: *mut trace_event_file = __data as *mut trace_event_file;
let mut __data_offsets: trace_event_data_offsets_ma_write = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut fbuffer: trace_event_buffer = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fbuffer) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 48usize);
}
let mut __data_size: i32 = 0;
if trace_trigger_soft_disabled(trace_file) {
return;
}
__data_size = trace_event_get_offsets_ma_write(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, r#fn, mas, piv, val);
let entry: *mut trace_event_raw_ma_write = (unsafe { trace_event_buffer_reserve(core::ptr::addr_of_mut!(fbuffer), trace_file, 72u64.wrapping_add(__data_size as u64)) }) as *mut trace_event_raw_ma_write;
if entry.is_null() {
return;
}
unsafe {
(*entry).r#fn = r#fn;
}
unsafe {
(*entry).min = (*mas).min;
}
unsafe {
(*entry).max = (*mas).max;
}
unsafe {
(*entry).index = (*mas).index;
}
unsafe {
(*entry).last = (*mas).last;
}
unsafe {
(*entry).piv = piv;
}
unsafe {
(*entry).val = val;
}
unsafe {
(*entry).node = (*mas).node as *mut ();
}
unsafe { trace_event_buffer_commit(core::ptr::addr_of_mut!(fbuffer)) };
}

#[unsafe(link_section = ".text.slate_distinct.maple_tree.trace_event_raw_event_ma_write")]
extern "C" fn trace_event_raw_event_ma_write(__data: *mut (), r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) {
let mut __UNIQUE_ID_guard_686: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_686) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_trace_event_raw_event_ma_write(__data, r#fn, mas, piv, val);
}

static mut print_fmt_ma_op: __SlateAlign16<[u8; 175]> = __SlateAlign16([34u8, 37u8, 115u8, 9u8, 78u8, 111u8, 100u8, 101u8, 58u8, 32u8, 37u8, 112u8, 32u8, 40u8, 37u8, 108u8, 117u8, 32u8, 37u8, 108u8, 117u8, 41u8, 32u8, 114u8, 97u8, 110u8, 103u8, 101u8, 58u8, 32u8, 37u8, 108u8, 117u8, 45u8, 37u8, 108u8, 117u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 102u8, 110u8, 44u8, 32u8, 40u8, 118u8, 111u8, 105u8, 100u8, 32u8, 42u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 110u8, 111u8, 100u8, 101u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 109u8, 105u8, 110u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 109u8, 97u8, 120u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 105u8, 110u8, 100u8, 101u8, 120u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 108u8, 97u8, 115u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_ma_op: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__maple_tree__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state)>, *mut ()>(Some(trace_event_raw_event_ma_op)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state)>, *mut ()>(Some(perf_trace_ma_op)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_ma_op.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_ma_op.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_ma_op.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_ma_op: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_ma_op) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_ma_op) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_ma_op) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_ma_op.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_ma_op: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_ma_op) };

static mut print_fmt_ma_read: __SlateAlign16<[u8; 175]> = __SlateAlign16([34u8, 37u8, 115u8, 9u8, 78u8, 111u8, 100u8, 101u8, 58u8, 32u8, 37u8, 112u8, 32u8, 40u8, 37u8, 108u8, 117u8, 32u8, 37u8, 108u8, 117u8, 41u8, 32u8, 114u8, 97u8, 110u8, 103u8, 101u8, 58u8, 32u8, 37u8, 108u8, 117u8, 45u8, 37u8, 108u8, 117u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 102u8, 110u8, 44u8, 32u8, 40u8, 118u8, 111u8, 105u8, 100u8, 32u8, 42u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 110u8, 111u8, 100u8, 101u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 109u8, 105u8, 110u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 109u8, 97u8, 120u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 105u8, 110u8, 100u8, 101u8, 120u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 108u8, 97u8, 115u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_ma_read: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__maple_tree__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state)>, *mut ()>(Some(trace_event_raw_event_ma_read)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state)>, *mut ()>(Some(perf_trace_ma_read)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_ma_read.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_ma_read.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_ma_read.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_ma_read: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_ma_read) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_ma_read) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_ma_read) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_ma_read.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_ma_read: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_ma_read) };

static mut print_fmt_ma_write: __SlateAlign16<[u8; 235]> = __SlateAlign16([34u8, 37u8, 115u8, 9u8, 78u8, 111u8, 100u8, 101u8, 32u8, 37u8, 112u8, 32u8, 40u8, 37u8, 108u8, 117u8, 32u8, 37u8, 108u8, 117u8, 41u8, 32u8, 114u8, 97u8, 110u8, 103u8, 101u8, 58u8, 37u8, 108u8, 117u8, 45u8, 37u8, 108u8, 117u8, 32u8, 112u8, 105u8, 118u8, 32u8, 40u8, 37u8, 108u8, 117u8, 41u8, 32u8, 118u8, 97u8, 108u8, 32u8, 37u8, 112u8, 34u8, 44u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 102u8, 110u8, 44u8, 32u8, 40u8, 118u8, 111u8, 105u8, 100u8, 32u8, 42u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 110u8, 111u8, 100u8, 101u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 109u8, 105u8, 110u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 109u8, 97u8, 120u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 105u8, 110u8, 100u8, 101u8, 120u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 108u8, 97u8, 115u8, 116u8, 44u8, 32u8, 40u8, 117u8, 110u8, 115u8, 105u8, 103u8, 110u8, 101u8, 100u8, 32u8, 108u8, 111u8, 110u8, 103u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 112u8, 105u8, 118u8, 44u8, 32u8, 40u8, 118u8, 111u8, 105u8, 100u8, 32u8, 42u8, 41u8, 32u8, 82u8, 69u8, 67u8, 45u8, 62u8, 118u8, 97u8, 108u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".ref.data")]
static mut event_class_ma_write: trace_event_class = {
let mut __t0: trace_event_class = unsafe { core::mem::zeroed() };
__t0.system = unsafe { core::ptr::addr_of!(str__maple_tree__trace_system_name) as *const u8 };
__t0.probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ())>, *mut ()>(Some(trace_event_raw_event_ma_write)) };
__t0.perf_probe = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut (), *const u8, *mut ma_state, u64, *mut ())>, *mut ()>(Some(perf_trace_ma_write)) };
__t0.reg = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call, u32, *mut ()) -> i32>>(trace_event_reg as *const ()) };
__t0.fields_array = unsafe { core::ptr::addr_of_mut!(trace_event_fields_ma_write.0) as *mut trace_event_fields };
__t0.fields = list_head { next: unsafe { core::ptr::addr_of_mut!(event_class_ma_write.fields) }, prev: unsafe { core::ptr::addr_of_mut!(event_class_ma_write.fields) } };
__t0.raw_init = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut trace_event_call) -> i32>>(trace_event_raw_init as *const ()) };
__t0
};

#[used(compiler)]
static mut event_ma_write: trace_event_call = {
let mut __t0: trace_event_call = unsafe { core::mem::zeroed() };
__t0.class = unsafe { core::ptr::addr_of_mut!(event_class_ma_write) };
__t0.__slate_anon_2 = __SlateRecord_trace_events_h_379 { tp: unsafe { core::ptr::addr_of_mut!(__tracepoint_ma_write) } };
__t0.event = {
let mut __t1: trace_event = unsafe { core::mem::zeroed() };
__t1.funcs = unsafe { core::ptr::addr_of_mut!(trace_event_type_funcs_ma_write) };
__t1
};
__t0.print_fmt = unsafe { core::ptr::addr_of_mut!(print_fmt_ma_write.0) as *mut u8 };
__t0.flags = 8i32;
__t0
};

#[used(compiler)]
#[unsafe(link_section = "_ftrace_events")]
static mut __event_ma_write: *mut trace_event_call = unsafe { core::ptr::addr_of_mut!(event_ma_write) };

fn do_perf_trace_ma_op(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_ma_op = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_ma_op(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, r#fn, mas);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr__7!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(56).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_ma_op = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_ma_op;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).r#fn = r#fn;
}
unsafe {
(*entry).min = (*mas).min;
}
unsafe {
(*entry).max = (*mas).max;
}
unsafe {
(*entry).index = (*mas).index;
}
unsafe {
(*entry).last = (*mas).last;
}
unsafe {
(*entry).node = (*mas).node as *mut ();
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.maple_tree.perf_trace_ma_op")]
extern "C" fn perf_trace_ma_op(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
let mut __UNIQUE_ID_guard_687: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_687) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_ma_op(__data, r#fn, mas);
}

fn do_perf_trace_ma_read(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_ma_read = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_ma_read(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, r#fn, mas);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr__7!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(56).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_ma_read = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_ma_read;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).r#fn = r#fn;
}
unsafe {
(*entry).min = (*mas).min;
}
unsafe {
(*entry).max = (*mas).max;
}
unsafe {
(*entry).index = (*mas).index;
}
unsafe {
(*entry).last = (*mas).last;
}
unsafe {
(*entry).node = (*mas).node as *mut ();
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.maple_tree.perf_trace_ma_read")]
extern "C" fn perf_trace_ma_read(__data: *mut (), r#fn: *const u8, mas: *mut ma_state) {
let mut __UNIQUE_ID_guard_688: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_688) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_ma_read(__data, r#fn, mas);
}

fn do_perf_trace_ma_write(__data: *mut (), r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) {
let event_call: *mut trace_event_call = __data as *mut trace_event_call;
let mut __data_offsets: trace_event_data_offsets_ma_write = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(__data_offsets) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 0usize);
}
let mut __regs: *mut pt_regs = unsafe { core::mem::zeroed() };
let __count: u64 = 1;
let __task: *mut task_struct = core::ptr::null_mut::<task_struct>();
let mut __entry_size: i32 = 0;
let __data_size: i32 = trace_event_get_offsets_ma_write(unsafe { &(*core::ptr::addr_of_mut!(__data_offsets)) }, r#fn, mas, piv, val);
let mut rctx: i32 = 0;
let mut __v0: *mut hlist_head;
__verify_pcpu_ptr__7!();
let mut __v2: *mut hlist_head;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*event_call).perf_events }) as u64);
__v2 = tcp_ptr__ as *mut hlist_head;
__v0 = __v2;
let head: *mut hlist_head = __v0;
let __v6: bool = !bpf_prog_array_valid(event_call) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d(__task.is_null()) } && __task.is_null() && hlist_empty(head as *const hlist_head) != 0;
if __v6 {
return;
}
__entry_size = ALIGN!((__data_size as u64).wrapping_add(72).wrapping_add(core::mem::size_of::<u32>() as u64), core::mem::size_of::<u64>() as u64) as i32;
__entry_size = (__entry_size as u64).wrapping_sub(core::mem::size_of::<u32>() as u64) as i32;
let entry: *mut trace_event_raw_ma_write = (unsafe { perf_trace_buf_alloc(__entry_size, core::ptr::addr_of_mut!(__regs), core::ptr::addr_of_mut!(rctx)) }) as *mut trace_event_raw_ma_write;
if entry.is_null() {
return;
}
perf_fetch_caller_regs(__regs);
unsafe {
(*entry).r#fn = r#fn;
}
unsafe {
(*entry).min = (*mas).min;
}
unsafe {
(*entry).max = (*mas).max;
}
unsafe {
(*entry).index = (*mas).index;
}
unsafe {
(*entry).last = (*mas).last;
}
unsafe {
(*entry).piv = piv;
}
unsafe {
(*entry).val = val;
}
unsafe {
(*entry).node = (*mas).node as *mut ();
}
unsafe { perf_trace_run_bpf_submit(entry as *mut (), __entry_size, rctx, event_call, __count, __regs, head, __task) };
}

#[unsafe(link_section = ".text.slate_distinct.maple_tree.perf_trace_ma_write")]
extern "C" fn perf_trace_ma_write(__data: *mut (), r#fn: *const u8, mas: *mut ma_state, piv: u64, val: *mut ()) {
let mut __UNIQUE_ID_guard_689: __SlateRecord_preempt_h_497 = class_preempt_notrace_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_689) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_preempt_notrace_destructor(__slate_cleanup_ptr_0 as *mut _));
do_perf_trace_ma_write(__data, r#fn, mas, piv, val);
}

// DOC: Interesting implementation details of the Maple Tree
//
// Each node type has a number of slots for entries and a number of slots for
// pivots.  In the case of dense nodes, the pivots are implied by the position
// and are simply the slot index + the minimum of the node.
//
// In regular B-Tree terms, pivots are called keys.  The term pivot is used to
// indicate that the tree is specifying ranges.  Pivots may appear in the
// subtree with an entry attached to the value whereas keys are unique to a
// specific position of a B-tree.  Pivot values are inclusive of the slot with
// the same index.
//
//
// The following illustrates the layout of a range64 nodes slots and pivots.
//
//
//  Slots -> | 0 | 1 | 2 | ... | 12 | 13 | 14 | 15 |
//           ┬   ┬   ┬   ┬     ┬    ┬    ┬    ┬    ┬
//           │   │   │   │     │    │    │    │    └─ Implied maximum
//           │   │   │   │     │    │    │    └─ Pivot 14
//           │   │   │   │     │    │    └─ Pivot 13
//           │   │   │   │     │    └─ Pivot 12
//           │   │   │   │     └─ Pivot 11
//           │   │   │   └─ Pivot 2
//           │   │   └─ Pivot 1
//           │   └─ Pivot 0
//           └─  Implied minimum
//
// Slot contents:
//  Internal (non-leaf) nodes contain pointers to other nodes.
//  Leaf nodes contain entries.
//
// The location of interest is often referred to as an offset.  All offsets have
// a slot, but the last offset has an implied pivot from the node above (or
// UINT_MAX for the root node.
//
// Ranges complicate certain write activities.  When modifying any of
// the B-tree variants, it is known that one entry will either be added or
// deleted.  When modifying the Maple Tree, one store operation may overwrite
// the entire data set, or one half of the tree, or the middle half of the tree.
// Kernel pointer hashing renders much of the maple tree dump useless as tagged
// pointers get hashed to arbitrary values.
//
// If CONFIG_DEBUG_VM_MAPLE_TREE is set we are in a debug mode where it is
// permissible to bypass this. Otherwise remain cautious and retain the hashing.
//
// Userland doesn't know about %px so also use %p there.
// Maple state flags
// * MA_STATE_PREALLOC		- Preallocated nodes, WARN_ON allocation
static mut maple_node_cache: *mut kmem_cache = unsafe { core::mem::zeroed() };

static mt_slots: [u8; 5] = [31, 16, 16, 10, 3];

static mt_pivots: [u8; 5] = [0, 15, 15, 9, 3];

/// Should never be used
static mt_min_slots: [u8; 5] = [15, 6, 6, 4, 1];

/// Functions
fn mt_alloc_one(gfp: u32) -> *mut maple_node {
alloc_hooks!(unsafe { kmem_cache_alloc_noprof(maple_node_cache, gfp) }) as *mut maple_node
}

fn mt_free_bulk(size: u64, nodes: *mut *mut ()) {
unsafe { kmem_cache_free_bulk(maple_node_cache, size, nodes) };
}

fn mt_return_sheaf(sheaf: *mut slab_sheaf) {
unsafe { kmem_cache_return_sheaf(maple_node_cache, (BIT!(11i32) as u32) | (BIT!(13i32) as u32), sheaf) };
}

fn mt_get_sheaf(gfp: u32, count: i32) -> *mut slab_sheaf {
unsafe { kmem_cache_prefill_sheaf(maple_node_cache, gfp, count as u32) }
}

fn mt_refill_sheaf(gfp: u32, sheaf: *mut *mut slab_sheaf, size: u32) -> i32 {
unsafe { kmem_cache_refill_sheaf(maple_node_cache, gfp, sheaf, size) }
}

// ma_free_rcu() - Use rcu callback to free a maple node
// @node: The node to free
//
// The maple tree uses the parent pointer to indicate this node is no longer in
// use and will be freed.
fn ma_free_rcu(node: *mut maple_node) {
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*node).__slate_anon_0.__slate_anon_0.parent }) != ma_parent_ptr!(node)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("690: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 690b - ., 3; .popsection\n# {0}", const 690 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ma_free_rcu_1, sym __slate_asm_string_ma_free_rcu_2, const 172 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("691: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 691b - ., 4; .popsection\n# {0}", const 691 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
loop {
let ___p: *mut maple_node = node;
if !___p.is_null() {
loop {
if offsetof__6!() >= 4096 {
unsafe { __compiletime_assert_692() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
let ___head: *mut kvfree_rcu_head = (unsafe { core::ptr::addr_of_mut!((*___p).__slate_anon_0.__slate_anon_1.rcu) }) as *mut kvfree_rcu_head;
unsafe { kvfree_call_rcu(___head, ___p as *mut ()) };
}
if true {
break;
}
}
}

fn mt_set_height(mt: *mut maple_tree, height: u8) {
let mut new_flags: u32 = unsafe { (*mt).ma_flags };
let __v1: u32 = new_flags & 4294967171u32;
new_flags = __v1;
loop {
if unlikely!((height as i32) > 31) != 0 {
loop {
unsafe {
core::arch::asm!("693: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 693b - ., 3; .popsection\n# {0}", const 693 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mt_set_height_2, sym __slate_asm_string_mt_set_height_3, const 181 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
new_flags |= ((height as i32) << 2i32) as u32;
unsafe {
(*mt).ma_flags = new_flags;
}
}

fn mas_mt_height(mas: *mut ma_state) -> u32 {
mt_height((unsafe { (*mas).tree }) as *const maple_tree)
}

fn mt_attr(mt: *mut maple_tree) -> u32 {
(unsafe { (*mt).ma_flags }) & 4294967171u32
}

#[inline(always)]
fn mte_node_type(entry: *const maple_enode) -> u32 {
((entry as u64) >> 3i32 & 15) as u32
}

#[inline(always)]
fn ma_is_dense(r#type: u32) -> bool {
(r#type as u32) < 1
}

#[inline(always)]
fn ma_is_leaf(r#type: u32) -> bool {
(r#type as u32) < 2
}

#[inline(always)]
fn mte_is_leaf(entry: *const maple_enode) -> bool {
ma_is_leaf(mte_node_type(entry))
}

/// We also reserve values with the bottom two bits set to '10' which are
/// below 4096
#[inline(always)]
fn mt_is_reserved(entry: *const ()) -> bool {
(entry as u64) < 4096 && xa_is_internal(entry)
}

#[inline(always)]
fn mas_set_err(mas: *mut ma_state, err: i64) {
unsafe {
(*mas).node = MA_ERROR!(err);
}
unsafe {
(*mas).status = 7;
}
}

#[inline(always)]
fn mas_is_ptr(mas: *const ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 2
}

#[inline(always)]
fn mas_is_start(mas: *const ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 1
}

#[inline(always)]
fn mas_is_none(mas: *const ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 3
}

#[inline(always)]
fn mas_is_paused(mas: *const ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 4
}

#[inline(always)]
fn mas_is_overflow(mas: *mut ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 5
}

fn mas_is_underflow(mas: *mut ma_state) -> bool {
((unsafe { (*mas).status }) as u32) == 6
}

fn mas_make_walkable(mas: *mut ma_state) {
let __v0: bool = !mas_is_active(mas) && !mas_is_start(mas as *const ma_state);
if __v0 {
unsafe {
(*mas).status = 1;
}
}
}

#[inline(always)]
fn mte_to_node(entry: *const maple_enode) -> *mut maple_node {
((entry as u64) & !255) as *mut maple_node
}

/// mte_to_mat() - Convert a maple encoded node to a maple topiary node.
/// @entry: The maple encoded node
///
/// Return: a maple topiary pointer
fn mte_to_mat(entry: *const maple_enode) -> *mut maple_topiary {
((entry as u64) & !255) as *mut maple_topiary
}

/// mas_mn() - Get the maple state node.
/// @mas: The maple state
///
/// Return: the maple node (not encoded - bare pointer).
fn mas_mn(mas: *const ma_state) -> *mut maple_node {
mte_to_node((unsafe { (*mas).node }) as *const maple_enode)
}

/// mte_set_node_dead() - Set a maple encoded node as dead.
/// @mn: The maple encoded node.
fn mte_set_node_dead(mn: *mut maple_enode) {
unsafe {
(*mte_to_node(mn as *const maple_enode)).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr!(mte_to_node(mn as *const maple_enode));
}
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
// Needed for RCU
}

// Bit 1 indicates the root is a node
// maple_type stored bit 3-6
// Bit 2 means a NULL somewhere below
fn mt_mk_node(node: *const maple_node, r#type: u32) -> *mut maple_enode {
(((node as u64) | (((r#type as u32) << 3i32) as u64) | 4) as *mut ()) as *mut maple_enode
}

fn ma_init_slot(slot: *mut *mut (), mn: *const maple_node, mt: u32) {
// WARNING: this is unsafe if the slot is exposed to readers.
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_694() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slot, RCU_INITIALIZER!(mt_mk_node(mn, mt) as *mut ()));
if true {
break;
}
}
}

fn mte_mk_root(node: *const maple_enode) -> *mut () {
((node as u64) | 2) as *mut ()
}

fn mte_safe_root(node: *const maple_enode) -> *mut () {
((node as u64) & 18446744073709551613u64) as *mut ()
}

fn mte_set_full(node: &maple_enode) -> *mut () {
(((node as *const maple_enode) as u64) & 18446744073709551611u64) as *mut ()
}

fn mte_clear_full(node: &maple_enode) -> *mut () {
(((node as *const maple_enode) as u64) | 4) as *mut ()
}

fn mte_has_null(node: &maple_enode) -> bool {
((node as *const maple_enode) as u64) & 4 != 0
}

#[inline(always)]
fn ma_is_root(node: *mut maple_node) -> bool {
((unsafe { (*node).__slate_anon_0.__slate_anon_0.parent }) as u64) & 1 != 0
}

#[inline(always)]
fn mte_is_root(node: *const maple_enode) -> bool {
ma_is_root(mte_to_node(node))
}

#[inline(always)]
fn mt_is_alloc(mt: *mut maple_tree) -> bool {
(unsafe { (*mt).ma_flags }) & 1 != 0
}

// The Parent Pointer
// Excluding root, the parent pointer is 256B aligned like all other tree nodes.
// When storing a 32 or 64 bit values, the offset can fit into 5 bits.  The 16
// bit values need an extra bit to store the offset.  This extra bit comes from
// a reuse of the last bit in the node type.  This is possible by using bit 1 to
// indicate if bit 2 is part of the type or the slot.
//
// Node types:
//  0b??1 = Root
//  0b?00 = 16 bit nodes
//  0b010 = 32 bit nodes
//  0b110 = 64 bit nodes
//
// Slot size and alignment
//  0b??1 : Root
//  0b?00 : 16 bit values, type in 0-1, slot in 2-7
//  0b010 : 32 bit values, type in 0-2, slot in 3-7
//  0b110 : 64 bit values, type in 0-2, slot in 3-7
/// mte_parent_shift() - Get the parent shift for the slot storage.
/// @parent: The parent pointer cast as an unsigned long
/// Return: The shift into that pointer to the star to of the slot
fn mte_parent_shift(parent: u64) -> u64 {
// Note bit 1 == 0 means 16B
if likely__2!(parent & 2) != 0 {
return 3;
}
2
}

/// mte_parent_slot_mask() - Get the slot mask for the parent.
/// @parent: The parent pointer cast as an unsigned long.
/// Return: The slot mask for that parent.
fn mte_parent_slot_mask(parent: u64) -> u64 {
// Note bit 1 == 0 means 16B
if likely__2!(parent & 2) != 0 {
return 248;
}
252
}

/// mas_parent_type() - Return the maple_type of the parent from the stored
/// parent type.
/// @mas: The maple state
/// @enode: The maple_enode to extract the parent's enum
/// Return: The node->parent maple_type
fn mas_parent_type(mas: *mut ma_state, enode: *mut maple_enode) -> u32 {
let mut p_type: u64 = (unsafe { (*mte_to_node(enode as *const maple_enode)).__slate_anon_0.__slate_anon_0.parent }) as u64;
let mut __v0: i64;
let __ret_warn_on: i32 = (p_type & 1 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("695: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 695b - ., 3; .popsection\n# {0}", const 695 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_parent_type_1, sym __slate_asm_string_mas_parent_type_2, const 444 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("696: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 696b - ., 4; .popsection\n# {0}", const 696 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 0;
}
p_type &= 255;
p_type &= !mte_parent_slot_mask(p_type);
match p_type {
6 => {
if mt_is_alloc(unsafe { (*mas).tree }) {
return 3;
}
// or MAPLE_PARENT_ARANGE64
return 2;
}
_ => {
}
}
0
}

/// mte_parent_slot() - get the parent slot of @enode.
/// @enode: The encoded maple node.
///
/// Return: The slot in the parent node where @enode resides.
#[inline(always)]
fn mte_parent_slot(enode: *const maple_enode) -> u32 {
let val: u64 = (unsafe { (*mte_to_node(enode)).__slate_anon_0.__slate_anon_0.parent }) as u64;
if unlikely__3!(val & 1) != 0 {
return 0;
}
// Okay to use MAPLE_PARENT_16B_SLOT_MASK as the last bit will be lost
// by shift if the parent shift is MAPLE_PARENT_SLOT_SHIFT
((val & 252) >> mte_parent_shift(val)) as u32
}

/// mte_parent() - Get the parent of @node.
/// @enode: The encoded maple node.
///
/// Return: The parent maple node.
#[inline(always)]
fn mte_parent(enode: *const maple_enode) -> *mut maple_node {
((((unsafe { (*mte_to_node(enode)).__slate_anon_0.__slate_anon_0.parent }) as u64) & !255) as *mut ()) as *mut maple_node
}

/// ma_dead_node() - check if the @enode is dead.
/// @enode: The encoded maple node
///
/// Return: true if dead, false otherwise.
#[inline(always)]
fn ma_dead_node(node: *const maple_node) -> bool {
// Do not reorder reads from the node prior to the parent check
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
let parent: *mut maple_node = ((((unsafe { (*node).__slate_anon_0.__slate_anon_0.parent }) as u64) & !255) as *mut ()) as *mut maple_node;
parent == (node as *mut maple_node)
}

/// mte_dead_node() - check if the @enode is dead.
/// @enode: The encoded maple node
///
/// Return: true if dead, false otherwise.
#[inline(always)]
fn mte_dead_node(enode: *const maple_enode) -> bool {
let node: *mut maple_node = mte_to_node(enode);
ma_dead_node(node as *const maple_node)
}

/// ma_pivots() - Get a pointer to the maple node pivots.
/// @node: the maple node
/// @type: the node type
///
/// In the event of a dead node, this array may be %NULL
///
/// Return: A pointer to the maple node pivots
fn ma_pivots(node: *mut maple_node, r#type: u32) -> *mut u64 {
match r#type as u32 {
3 => {
return unsafe { (*node).__slate_anon_0.ma64.pivot.as_mut_ptr() as *mut u64 };
}
1 | 2 => {
return unsafe { (*node).__slate_anon_0.mr64.pivot.as_mut_ptr() as *mut u64 };
}
4 => {
return unsafe { (*node).__slate_anon_0.cp.__slate_anon_5.pivot.as_mut_ptr() as *mut u64 };
}
0 => {
return core::ptr::null_mut::<u64>();
}
_ => {
}
}
core::ptr::null_mut::<u64>()
}

/// ma_gaps() - Get a pointer to the maple node gaps.
/// @node: the maple node
/// @type: the node type
///
/// Return: A pointer to the maple node gaps
fn ma_gaps(node: *mut maple_node, r#type: u32) -> *mut u64 {
match r#type as u32 {
3 => {
return unsafe { (*node).__slate_anon_0.ma64.gap.as_mut_ptr() as *mut u64 };
}
4 => {
return unsafe { (*node).__slate_anon_0.cp.gap.as_mut_ptr() as *mut u64 };
}
0..=2 => {
return core::ptr::null_mut::<u64>();
}
_ => {
}
}
core::ptr::null_mut::<u64>()
}

/// mas_safe_pivot() - get the pivot at @piv or mas->max.
/// @mas: The maple state
/// @pivots: The pointer to the maple node pivots
/// @piv: The pivot to fetch
/// @type: The maple node type
///
/// Return: The pivot at @piv within the limit of the @pivots array, @mas->max
/// otherwise.
#[inline(always)]
fn mas_safe_pivot(mas: *const ma_state, pivots: *mut u64, piv: u8, r#type: u32) -> u64 {
if (piv as i32) >= ((unsafe { mt_pivots[(r#type as u32) as usize] }) as i32) {
return unsafe { (*mas).max };
}
unsafe { *pivots.wrapping_offset((piv as i32) as isize) }
}

/// mas_safe_min() - Return the minimum for a given offset.
/// @mas: The maple state
/// @pivots: The pointer to the maple node pivots
/// @offset: The offset into the pivot array
///
/// Return: The minimum range value that is contained in @offset.
fn mas_safe_min(mas: *mut ma_state, pivots: *mut u64, offset: u8) -> u64 {
if likely__5!(offset) != 0 {
return unsafe { *pivots.wrapping_offset((offset as i32).wrapping_sub(1) as isize) }.wrapping_add(1);
}
unsafe { (*mas).min }
}

/// mte_set_pivot() - Set a pivot to a value in an encoded maple node.
/// @mn: The encoded maple node
/// @piv: The pivot offset
/// @val: The value of the pivot
fn mte_set_pivot(mn: *mut maple_enode, piv: u8, val: u64) {
let node: *mut maple_node = mte_to_node(mn as *const maple_enode);
let r#type: u32 = mte_node_type(mn as *const maple_enode);
loop {
if unlikely!((piv as i32) >= ((unsafe { mt_pivots[(r#type as u32) as usize] }) as i32)) != 0 {
loop {
unsafe {
core::arch::asm!("697: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 697b - ., 3; .popsection\n# {0}", const 697 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mte_set_pivot_0, sym __slate_asm_string_mte_set_pivot_1, const 621 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
match r#type as u32 {
1 | 2 => {
unsafe {
*((*node).__slate_anon_0.mr64.pivot.as_mut_ptr() as *mut u64).wrapping_offset((piv as i32) as isize) = val;
}
}
3 => {
unsafe {
*((*node).__slate_anon_0.ma64.pivot.as_mut_ptr() as *mut u64).wrapping_offset((piv as i32) as isize) = val;
}
}
0 | 4 => {
}
_ => {
}
}
}

/// ma_slots() - Get a pointer to the maple node slots.
/// @mn: The maple node
/// @mt: The maple node type
///
/// Return: A pointer to the maple node slots
fn ma_slots(mn: *mut maple_node, mt: u32) -> *mut *mut () {
match mt as u32 {
3 => {
return unsafe { (*mn).__slate_anon_0.ma64.slot.as_mut_ptr() as *mut *mut () };
}
1 | 2 => {
return unsafe { (*mn).__slate_anon_0.mr64.__slate_anon_2.slot.as_mut_ptr() as *mut *mut () };
}
4 => {
return unsafe { (*mn).__slate_anon_0.cp.slot.as_mut_ptr() as *mut *mut () };
}
0 => {
return unsafe { (*mn).__slate_anon_0.__slate_anon_0.slot.as_mut_ptr() as *mut *mut () };
}
_ => {
}
}
core::ptr::null_mut::<*mut ()>()
}

fn mt_write_locked(mt: *const maple_tree) -> bool {
let __v0: i32 = if mt_external_lock(mt) {
mt_write_lock_is_held!()
} else {
unsafe { lockdep_is_held(core::ptr::addr_of!((*mt).__slate_anon_0.ma_lock) as *const ()) }
};
__v0 != 0
}

#[inline(always)]
fn mt_locked(mt: *const maple_tree) -> bool {
let __v0: i32 = if mt_external_lock(mt) {
mt_lock_is_held!()
} else {
unsafe { lockdep_is_held(core::ptr::addr_of!((*mt).__slate_anon_0.ma_lock) as *const ()) }
};
__v0 != 0
}

#[inline(always)]
fn mt_slot(mt: *const maple_tree, slots: *mut *mut (), offset: u8) -> *mut () {
let mut __v0: *mut ();
let mut __v1: *mut ();
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_699() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(slots.wrapping_offset((offset as i32) as isize) as *const *mut ()) };
let __UNIQUE_ID_rcu_698: *mut () = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = mt_locked(mt) || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_698;
__v0
}

#[inline(always)]
fn mt_slot_locked(mt: *mut maple_tree, slots: *mut *mut (), offset: u8) -> *mut () {
rcu_dereference_protected__5!(unsafe { *slots.wrapping_offset((offset as i32) as isize) }, mt_write_locked(mt as *const maple_tree))
}

/// mas_slot_locked() - Get the slot value when holding the maple tree lock.
/// @mas: The maple state
/// @slots: The pointer to the slots
/// @offset: The offset into the slots array to fetch
///
/// Return: The entry stored in @slots at the @offset.
#[inline(always)]
fn mas_slot_locked(mas: *mut ma_state, slots: *mut *mut (), offset: u8) -> *mut () {
mt_slot_locked(unsafe { (*mas).tree }, slots, offset)
}

/// mas_slot() - Get the slot value when not holding the maple tree lock.
/// @mas: The maple state
/// @slots: The pointer to the slots
/// @offset: The offset into the slots array to fetch
///
/// Return: The entry stored in @slots at the @offset
#[inline(always)]
fn mas_slot(mas: *mut ma_state, slots: *mut *mut (), offset: u8) -> *mut () {
mt_slot((unsafe { (*mas).tree }) as *const maple_tree, slots, offset)
}

// mas_root() - Get the maple tree root.
// @mas: The maple state.
//
// Return: The pointer to the root of the tree
#[inline(always)]
fn mas_root(mas: *mut ma_state) -> *mut () {
let mut __v0: *mut ();
let mut __v1: *mut ();
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_701() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*mas).tree).ma_root) as *const *mut ()) };
let __UNIQUE_ID_rcu_700: *mut () = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = mt_locked((unsafe { (*mas).tree }) as *const maple_tree) || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_700;
__v0
}

fn mt_root_locked(mt: *mut maple_tree) -> *mut () {
rcu_dereference_protected__5!(unsafe { (*mt).ma_root }, mt_write_locked(mt as *const maple_tree))
}

/// mas_root_locked() - Get the maple tree root when holding the maple tree lock.
/// @mas: The maple state.
///
/// Return: The pointer to the root of the tree
fn mas_root_locked(mas: *mut ma_state) -> *mut () {
mt_root_locked(unsafe { (*mas).tree })
}

fn ma_meta(mn: *mut maple_node, mt: u32) -> *mut maple_metadata {
match mt as u32 {
3 => {
return unsafe { core::ptr::addr_of_mut!((*mn).__slate_anon_0.ma64.meta) };
}
_ => {
return unsafe { core::ptr::addr_of_mut!((*mn).__slate_anon_0.mr64.__slate_anon_2.__slate_anon_1.meta) };
}
}
unsafe { core::mem::zeroed() }
}

/// ma_set_meta() - Set the metadata information of a node.
/// @mn: The maple node
/// @mt: The maple node type
/// @offset: The offset of the highest sub-gap in this node.
/// @end: The end of the data in this node.
fn ma_set_meta(mn: *mut maple_node, mt: u32, offset: u8, end: u8) {
let meta: *mut maple_metadata = ma_meta(mn, mt);
unsafe {
(*meta).gap = offset;
}
unsafe {
(*meta).end = end;
}
}

/// mt_clear_meta() - clear the metadata information of a node, if it exists
/// @mt: The maple tree
/// @mn: The maple node
/// @type: The maple node type
fn mt_clear_meta(mt: *mut maple_tree, mn: *mut maple_node, r#type: u32) {
let mut meta: *mut maple_metadata = unsafe { core::mem::zeroed() };
// no metadata, could be node
match r#type as u32 {
2 => {
let pivots: *mut u64 = unsafe { (*mn).__slate_anon_0.mr64.pivot.as_mut_ptr() as *mut u64 };
if unlikely__3!(unsafe { *pivots.wrapping_offset(14i32 as isize) }) != 0 {
let slots: *mut *mut () = unsafe { (*mn).__slate_anon_0.mr64.__slate_anon_2.slot.as_mut_ptr() as *mut *mut () };
let next: *mut () = mt_slot_locked(mt, slots, 15u8);
let __v1: bool = !mte_to_node(next as *const maple_enode).is_null() && (mte_node_type(next as *const maple_enode) as u32) != 0;
if __v1 {
return;
}
// no metadata, could be node
}
meta = ma_meta(mn, r#type);
}
3 => {
meta = ma_meta(mn, r#type);
}
_ => {
return;
}
}
unsafe {
(*meta).gap = 0;
}
unsafe {
(*meta).end = 0;
}
}

/// ma_meta_end() - Get the data end of a node from the metadata
/// @mn: The maple node
/// @mt: The maple node type
fn ma_meta_end(mn: *mut maple_node, mt: u32) -> u8 {
let meta: *mut maple_metadata = ma_meta(mn, mt);
unsafe { (*meta).end }
}

/// ma_meta_gap() - Get the largest gap location of a node from the metadata
/// @mn: The maple node
fn ma_meta_gap(mn: *mut maple_node) -> u8 {
unsafe { (*mn).__slate_anon_0.ma64.meta.gap }
}

/// ma_set_meta_gap() - Set the largest gap location in a nodes metadata
/// @mn: The maple node
/// @mt: The maple node type
/// @offset: The location of the largest gap.
fn ma_set_meta_gap(mn: *mut maple_node, mt: u32, offset: u8) {
let meta: *mut maple_metadata = ma_meta(mn, mt);
unsafe {
(*meta).gap = offset;
}
}

/// mas_set_parent_slots() - Bulk operation to set many slot parent pointers
/// @mas: The maple state
/// @parent: The encoded maple node that is the parent of @enode.
/// @slot: The slot that of the @enode.
/// @start_slot: The offset into @slot
/// @count: The number of slots to set (eg: exclusive)
fn mas_set_parent_slots(mas: *mut ma_state, parent: *mut maple_enode, slots: *mut *mut (), start_slot: u8, count: u8) {
let mut val: u64 = 0;
let shift: u64 = 3;
let r#type: u64 = 6;
let p_type: u32 = mte_node_type(parent as *const maple_enode);
let mut i: u8 = 0;
loop {
if unlikely!((p_type as u32) != 2 && (p_type as u32) != 3) != 0 {
loop {
unsafe {
core::arch::asm!("702: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 702b - ., 3; .popsection\n# {0}", const 702 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mas_set_parent_slots_0, sym __slate_asm_string_mas_set_parent_slots_1, const 859 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
val = parent as u64;
let __v6: u64 = val & !255;
val = __v6;
while (i as i32) < (count as i32) {
let pval: u64 = val | (((start_slot as i32).wrapping_add(i as i32) << shift) as u64) | r#type;
let child: *mut maple_enode = mt_slot_locked(unsafe { (*mas).tree }, slots, i) as *mut maple_enode;
unsafe {
(*mte_to_node(child as *const maple_enode)).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr!(pval);
}
i = (i as i32).wrapping_add(1) as u8;
}
}

/// mat_add() - Add a @dead_enode to the ma_topiary of a list of dead nodes.
/// @mat: the ma_topiary, a linked list of dead nodes.
/// @dead_enode: the node to be marked as dead and added to the tail of the list
///
/// Add the @dead_enode to the linked list in @mat.
fn mat_add(mat: &mut ma_topiary, dead_enode: *mut maple_enode) {
mte_set_node_dead(dead_enode);
unsafe {
(*mte_to_mat(dead_enode as *const maple_enode)).next = core::ptr::null_mut::<maple_enode>();
}
if unsafe { (*(mat as *mut ma_topiary)).tail }.is_null() {
let __v0: *mut maple_enode = dead_enode;
unsafe {
(*(mat as *mut ma_topiary)).head = __v0;
}
unsafe {
(*(mat as *mut ma_topiary)).tail = __v0;
}
return;
}
unsafe {
(*mte_to_mat((*(mat as *mut ma_topiary)).tail as *const maple_enode)).next = dead_enode;
}
unsafe {
(*(mat as *mut ma_topiary)).tail = dead_enode;
}
}

/// mas_mat_destroy() - Free all nodes and subtrees in a dead list.
/// @mas: the maple state
/// @mat: the ma_topiary linked list of dead nodes to free.
///
/// Destroy walk a dead list.
fn mas_mat_destroy(mas: *mut ma_state, mat: &mut ma_topiary) {
let mut node: *mut maple_node = unsafe { core::mem::zeroed() };
let in_rcu: bool = mt_in_rcu(unsafe { (*mas).tree });
while !unsafe { (*(mat as *mut ma_topiary)).head }.is_null() {
let next: *mut maple_enode = unsafe { (*mte_to_mat((*(mat as *mut ma_topiary)).head as *const maple_enode)).next };
node = mte_to_node((unsafe { (*(mat as *mut ma_topiary)).head }) as *const maple_enode);
mt_destroy_walk(unsafe { (*(mat as *mut ma_topiary)).head }, unsafe { (*mas).tree }, !in_rcu);
if in_rcu {
unsafe { call_rcu(core::ptr::addr_of_mut!((*node).__slate_anon_0.__slate_anon_1.rcu), Some(mt_free_walk)) };
}
unsafe {
(*(mat as *mut ma_topiary)).head = next;
}
}
}

/// mas_descend() - Descend into the slot stored in the ma_state.
/// @mas: the maple state.
///
/// Note: Not RCU safe, only use in write side or debug code.
fn mas_descend(mas: *mut ma_state) {
let mut r#type: u32 = 0;
let node: *mut maple_node = mas_mn(mas as *const ma_state);
r#type = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
let pivots: *mut u64 = ma_pivots(node, r#type);
let slots: *mut *mut () = ma_slots(node, r#type);
if (unsafe { (*mas).offset }) != 0 {
unsafe {
(*mas).min = unsafe { *pivots.wrapping_offset(((*mas).offset as i32).wrapping_sub(1) as isize) }.wrapping_add(1);
}
}
unsafe {
(*mas).max = mas_safe_pivot(mas as *const ma_state, pivots, (*mas).offset, r#type);
}
unsafe {
(*mas).node = mas_slot(mas, slots, (*mas).offset) as *mut maple_enode;
}
}

/// mas_ascend() - Walk up a level of the tree.
/// @mas: The maple state
///
/// Sets the @mas->max and @mas->min for the parent node of mas->node.  This
/// may cause several levels of walking up to find the correct min and max.
/// May find a dead node which will cause a premature return.
/// Return: 1 on dead node, 0 otherwise
fn mas_ascend(mas: *mut ma_state) -> i32 {
let mut a_enode: *mut maple_enode = unsafe { core::mem::zeroed() }; // ancestor enode.
let mut a_node: *mut maple_node = unsafe { core::mem::zeroed() }; // ancestor node.
let mut a_slot: u8 = 0;
let mut a_type: u32 = 0;
let mut min: u64 = 0;
let mut max: u64 = !0;
let mut set_max: bool = false;
let mut set_min: bool = false;
a_node = mas_mn(mas as *const ma_state);
if ma_is_root(a_node) {
unsafe {
(*mas).offset = 0;
}
return 0;
}
// parent node.
let p_node: *mut maple_node = mte_parent((unsafe { (*mas).node }) as *const maple_enode);
if unlikely!(a_node == p_node) != 0 {
return 1;
}
a_type = mas_parent_type(mas, unsafe { (*mas).node });
unsafe {
(*mas).offset = mte_parent_slot((*mas).node as *const maple_enode) as u8;
}
a_enode = mt_mk_node(p_node as *const maple_node, a_type);
// Check to make sure all parent information is still accurate
if p_node != mte_parent((unsafe { (*mas).node }) as *const maple_enode) {
return 1;
}
unsafe {
(*mas).node = a_enode;
}
if mte_is_root(a_enode as *const maple_enode) {
unsafe {
(*mas).max = !0;
}
unsafe {
(*mas).min = 0;
}
return 0;
}
// !mas->offset implies that parent node min == mas->min.
// mas->offset > 0 implies that we need to walk up to find the
// implied pivot min.
if (unsafe { (*mas).offset }) == 0 {
min = unsafe { (*mas).min };
set_min = true;
}
if (unsafe { (*mas).max }) == !0 {
set_max = true;
}
loop {
// parent enode.
let p_enode: *mut maple_enode = a_enode;
a_type = mas_parent_type(mas, p_enode);
a_node = mte_parent(p_enode as *const maple_enode);
a_slot = mte_parent_slot(p_enode as *const maple_enode) as u8;
a_enode = mt_mk_node(a_node as *const maple_node, a_type);
let pivots: *mut u64 = ma_pivots(a_node, a_type);
if unlikely!(ma_dead_node(a_node as *const maple_node)) != 0 {
return 1;
}
if !set_min && a_slot != 0 {
set_min = true;
min = unsafe { *pivots.wrapping_offset((a_slot as i32).wrapping_sub(1) as isize) }.wrapping_add(1);
}
if !set_max && (a_slot as i32) < ((unsafe { mt_pivots[(a_type as u32) as usize] }) as i32) {
set_max = true;
max = unsafe { *pivots.wrapping_offset((a_slot as i32) as isize) };
}
if unlikely!(ma_dead_node(a_node as *const maple_node)) != 0 {
return 1;
}
if unlikely!(ma_is_root(a_node)) != 0 {
break;
}
if !(!set_min || !set_max) {
break;
}
}
unsafe {
(*mas).max = max;
}
unsafe {
(*mas).min = min;
}
0
}

/// mas_pop_node() - Get a previously allocated maple node from the maple state.
/// @mas: The maple state
///
/// Return: A pointer to a maple node.
#[inline(always)]
fn mas_pop_node(mas: *mut ma_state) -> *mut maple_node {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut maple_node = core::ptr::addr_of_mut!(ret) as *mut *mut maple_node;
unsafe {
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if !unsafe { (*mas).alloc }.is_null() {
*__slate_slot_3 = (*mas).alloc;
(*mas).alloc = core::ptr::null_mut::<maple_node>();
} else {
__ret_warn_on = unsafe { (*mas).sheaf }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("703: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 703b - ., 3; .popsection\n# {0}", const 703 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_pop_node_9, sym __slate_asm_string_mas_pop_node_10, const 1059 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("704: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 704b - ., 4; .popsection\n# {0}", const 704 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return core::ptr::null_mut::<maple_node>();
} else {
*__slate_slot_3 = alloc_hooks!(kmem_cache_alloc_from_sheaf_noprof(maple_node_cache, 10240u32, (*mas).sheaf)) as *mut maple_node;
}
}
memset(*__slate_slot_3 as *mut (), 0i32, core::mem::size_of::<maple_node>() as u64);
return *__slate_slot_3;
}
}

/// mas_alloc_nodes() - Allocate nodes into a maple state
/// @mas: The maple state
/// @gfp: The GFP Flags
fn mas_alloc_nodes(mas: *mut ma_state, gfp: u32) {
let mut refill: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u64 = core::ptr::addr_of_mut!(refill) as *mut u64;
unsafe {
if (*mas).node_request == 0 {
return;
} else {
'__join_0: {
if (*mas).node_request == 1 {
if unsafe { (*mas).sheaf }.is_null() {
if !unsafe { (*mas).alloc }.is_null() {
return;
} else {
(*mas).alloc = mt_alloc_one(gfp);
if unsafe { (*mas).alloc }.is_null() {
break '__join_0;
} else {
(*mas).node_request = 0;
return;
}
}
}
}
if unlikely__6!(unsafe { (*mas).alloc }) != 0 {
kfree((*mas).alloc as *const ());
(*mas).alloc = core::ptr::null_mut::<maple_node>();
}
if !unsafe { (*mas).sheaf }.is_null() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = (*mas).node_request;
if (kmem_cache_sheaf_size((*mas).sheaf) as u64) >= *__slate_slot_0 {
(*mas).node_request = 0;
return;
} else {
if mt_refill_sheaf(gfp, core::ptr::addr_of_mut!((*mas).sheaf), *__slate_slot_0 as u32) == 0 {
(*mas).node_request = 0;
return;
}
}
} else {
(*mas).sheaf = mt_get_sheaf(gfp, (*mas).node_request as i32);
if likely__8!(unsafe { (*mas).sheaf }) != 0 {
(*mas).node_request = 0;
return;
}
}
}
mas_set_err(mas, 12i32.wrapping_neg() as i64);
}
}
}

fn mas_empty_nodes(mas: *mut ma_state) {
unsafe {
(*mas).node_request = 0;
}
if !unsafe { (*mas).sheaf }.is_null() {
mt_return_sheaf(unsafe { (*mas).sheaf });
unsafe {
(*mas).sheaf = core::ptr::null_mut::<slab_sheaf>();
}
}
if !unsafe { (*mas).alloc }.is_null() {
unsafe { kfree((*mas).alloc as *const ()) };
unsafe {
(*mas).alloc = core::ptr::null_mut::<maple_node>();
}
}
}

/// mas_free() - Free an encoded maple node
/// @mas: The maple state
/// @used: The encoded maple node to free.
///
/// Uses rcu free if necessary, pushes @used back on the maple state allocations
/// otherwise.
fn mas_free(mas: *mut ma_state, used: *mut maple_enode) {
ma_free_rcu(mte_to_node(used as *const maple_enode));
}

fn mas_lock_check(mas: *mut ma_state) {
}

fn mas_init_lock_check(mas: *mut ma_state) {
}

fn mas_may_init_lock_check(mas: *mut ma_state) {
}

/// mas_start() - Sets up maple state for operations.
/// @mas: The maple state.
///
/// If mas->status == ma_start, then set the min, max and depth to
/// defaults.
///
/// Return:
/// - If mas->node is an error or not mas_start, return NULL.
/// - If it's an empty tree:     NULL & mas->status == ma_none
/// - If it's a single entry:    The entry & mas->status == ma_root
/// - If it's a tree:            NULL & mas->status == ma_active
fn mas_start(mas: *mut ma_state) -> *mut maple_enode {
let mut root: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut maple_enode = core::ptr::addr_of_mut!(root) as *mut *mut maple_enode;
unsafe {
if likely!(mas_is_start(mas as *const ma_state)) != 0 {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
mas_init_lock_check(mas);
(*mas).min = 0;
(*mas).max = !0;
loop {
(*mas).depth = 0;
*__slate_slot_0 = mas_root(mas) as *mut maple_enode;
// Tree with nodes
if likely!(xa_is_node(*__slate_slot_0 as *const ())) != 0 {
(*mas).depth = 0;
(*mas).status = 0;
(*mas).node = mte_safe_root(*__slate_slot_0 as *const maple_enode) as *mut maple_enode;
(*mas).offset = 0;
if !mte_dead_node((*mas).node as *const maple_enode) {
return core::ptr::null_mut::<maple_enode>();
}
} else {
break;
}
}
(*mas).node = core::ptr::null_mut::<maple_enode>();
// empty tree
if unlikely!((*__slate_slot_0).is_null()) != 0 {
(*mas).status = 3;
(*mas).offset = 31;
return core::ptr::null_mut::<maple_enode>();
} else {
// Single entry tree
(*mas).status = 2;
(*mas).offset = 31;
// Single entry tree.
if (*mas).index > 0 {
return core::ptr::null_mut::<maple_enode>();
} else {
return *__slate_slot_0;
}
}
} else {
return core::ptr::null_mut::<maple_enode>();
}
}
}

/// ma_data_end() - Find the end of the data in a node.
/// @node: The maple node
/// @type: The maple node type
/// @pivots: The array of pivots in the node
/// @max: The maximum value in the node
///
/// Uses metadata to find the end of the data when possible.
/// Return: The zero indexed last slot with data (may be null).
#[inline(always)]
fn ma_data_end(node: *mut maple_node, r#type: u32, pivots: *mut u64, max: u64) -> u8 {
let mut offset: u8 = 0;
if pivots.is_null() {
return 0;
}
if (r#type as u32) == 3 {
return ma_meta_end(node, r#type);
}
offset = ((unsafe { mt_pivots[(r#type as u32) as usize] }) as i32).wrapping_sub(1) as u8;
if likely!((unsafe { *pivots.wrapping_offset((offset as i32) as isize) }) == 0) != 0 {
return ma_meta_end(node, r#type);
}
if likely!((unsafe { *pivots.wrapping_offset((offset as i32) as isize) }) == max) != 0 {
return offset;
}
unsafe { mt_pivots[(r#type as u32) as usize] }
}

/// mas_data_end() - Find the end of the data (slot).
/// @mas: the maple state
///
/// This method is optimized to check the metadata of a node if the node type
/// supports data end metadata.
///
/// Return: The zero indexed last slot with data (may be null).
fn mas_data_end(mas: *mut ma_state) -> u8 {
let mut r#type: u32 = 0;
let mut offset: u8 = 0;
r#type = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
let node: *mut maple_node = mas_mn(mas as *const ma_state);
if (r#type as u32) == 3 {
return ma_meta_end(node, r#type);
}
let pivots: *mut u64 = ma_pivots(node, r#type);
if unlikely!(ma_dead_node(node as *const maple_node)) != 0 {
return 0;
}
offset = ((unsafe { mt_pivots[(r#type as u32) as usize] }) as i32).wrapping_sub(1) as u8;
if likely!((unsafe { *pivots.wrapping_offset((offset as i32) as isize) }) == 0) != 0 {
return ma_meta_end(node, r#type);
}
if likely!((unsafe { *pivots.wrapping_offset((offset as i32) as isize) }) == unsafe { (*mas).max }) != 0 {
return offset;
}
unsafe { mt_pivots[(r#type as u32) as usize] }
}

fn wr_mas_setup(wr_mas: *mut ma_wr_state, mas: *mut ma_state) {
unsafe {
(*wr_mas).node = mas_mn(mas as *const ma_state);
}
unsafe {
(*wr_mas).r#type = mte_node_type((*mas).node as *const maple_enode);
}
unsafe {
(*wr_mas).pivots = ma_pivots((*wr_mas).node, (*wr_mas).r#type);
}
unsafe {
(*wr_mas).slots = ma_slots((*wr_mas).node, (*wr_mas).r#type);
}
unsafe {
(*wr_mas).r_min = mas_safe_min(mas, (*wr_mas).pivots, (*mas).offset);
}
unsafe {
(*wr_mas).r_max = mas_safe_pivot(mas as *const ma_state, (*wr_mas).pivots, (*mas).offset, (*wr_mas).r#type);
}
}

fn wr_mas_ascend(wr_mas: *mut ma_wr_state) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
mas_ascend(mas);
wr_mas_setup(wr_mas, mas);
unsafe {
(*mas).end = ma_data_end((*wr_mas).node, (*wr_mas).r#type, (*wr_mas).pivots, (*mas).max);
}
// Careful, this may be wrong..
unsafe {
(*wr_mas).end_piv = (*wr_mas).r_max;
}
unsafe {
(*wr_mas).offset_end = (*mas).offset;
}
}

fn ma_leaf_max_gap(mn: *mut maple_node, mt: u32, min: u64, max: u64, pivots: *mut u64, slots: *mut *mut ()) -> u64 {
let mut pstart: u64 = 0;
let mut gap: u64 = 0;
let mut max_gap: u64 = 0;
let mut i: u8 = 0;
let mut max_piv: u8 = 0;
if unlikely!(ma_is_dense(mt)) != 0 {
gap = 0;
i = 0;
while (i as i32) < ((unsafe { mt_slots[(mt as u32) as usize] }) as i32) {
if !unsafe { *slots.wrapping_offset((i as i32) as isize) }.is_null() {
if gap > max_gap {
max_gap = gap;
}
gap = 0;
} else {
gap = gap.wrapping_add(1);
}
i = (i as i32).wrapping_add(1) as u8;
}
if gap > max_gap {
max_gap = gap;
}
return max_gap;
}
// Check the first implied pivot optimizes the loop below and slot 1 may
// be skipped if there is a gap in slot 0.
if likely!(unsafe { *slots.wrapping_offset(0 as isize) }.is_null()) != 0 {
max_gap = unsafe { *pivots.wrapping_offset(0 as isize) }.wrapping_sub(min).wrapping_add(1);
i = 2;
} else {
i = 1;
}
// reduce max_piv as the special case is checked before the loop
max_piv = (ma_data_end(mn, mt, pivots, max) as i32).wrapping_sub(1) as u8;
// Check end implied pivot which can only be a gap on the right most
// node.
if unlikely!(max == !0) != 0 && unsafe { *slots.wrapping_offset((max_piv as i32).wrapping_add(1) as isize) }.is_null() {
gap = 18446744073709551615u64.wrapping_sub(unsafe { *pivots.wrapping_offset((max_piv as i32) as isize) });
if gap > max_gap {
max_gap = gap;
}
if max_gap > unsafe { *pivots.wrapping_offset((max_piv as i32) as isize) }.wrapping_sub(min) {
return max_gap;
}
}
while (i as i32) <= (max_piv as i32) {
// data == no gap.
if likely__4!(unsafe { *slots.wrapping_offset((i as i32) as isize) }) == 0 {
pstart = unsafe { *pivots.wrapping_offset((i as i32).wrapping_sub(1) as isize) };
gap = unsafe { *pivots.wrapping_offset((i as i32) as isize) }.wrapping_sub(pstart);
if gap > max_gap {
max_gap = gap;
}
// There cannot be two gaps in a row.
i = (i as i32).wrapping_add(1) as u8;
}
i = (i as i32).wrapping_add(1) as u8;
}
max_gap
}

/// mas_leaf_max_gap() - Returns the largest gap in a leaf node
/// @mas: the maple state
///
/// Return: The maximum gap in the leaf.
fn mas_leaf_max_gap(mas: *mut ma_state) -> u64 {
let mut mt: u32 = 0;
let mn: *mut maple_node = mas_mn(mas as *const ma_state);
mt = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
let slots: *mut *mut () = ma_slots(mn, mt);
let pivots: *mut u64 = ma_pivots(mn, mt);
ma_leaf_max_gap(mn, mt, unsafe { (*mas).min }, unsafe { (*mas).max }, pivots, slots)
}

/// ma_max_gap() - Get the maximum gap in a maple node (non-leaf)
/// @node: The maple node
/// @gaps: The pointer to the gaps
/// @mt: The maple node type
/// @off: Pointer to store the offset location of the gap.
///
/// Uses the metadata data end to scan backwards across set gaps.
///
/// Return: The maximum gap value
fn ma_max_gap(node: *mut maple_node, gaps: *mut u64, mt: u32, off: &mut u8) -> u64 {
let mut offset: u8 = 0;
let mut i: u8 = 0;
let mut max_gap: u64 = 0;
let __v0: u8 = ma_meta_end(node, mt);
offset = __v0;
i = __v0;
loop {
if (unsafe { *gaps.wrapping_offset((i as i32) as isize) }) > max_gap {
max_gap = unsafe { *gaps.wrapping_offset((i as i32) as isize) };
offset = i;
}
let __v2: u8 = i;
i = (__v2 as i32).wrapping_sub(1) as u8;
if __v2 == 0 {
break;
}
}
unsafe {
*(off as *mut u8) = offset;
}
max_gap
}

/// mas_max_gap() - find the largest gap in a non-leaf node and set the slot.
/// @mas: The maple state.
///
/// Return: The gap value.
fn mas_max_gap(mas: *mut ma_state) -> u64 {
let mut offset: u8 = 0;
let mut mt: u32 = 0;
mt = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
if ma_is_leaf(mt) {
return mas_leaf_max_gap(mas);
}
let node: *mut maple_node = mas_mn(mas as *const ma_state);
loop {
if unlikely!((mt as u32) != 3) != 0 {
loop {
unsafe {
core::arch::asm!("705: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 705b - ., 3; .popsection\n# {0}", const 705 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mas_max_gap_0, sym __slate_asm_string_mas_max_gap_1, const 1513 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
offset = ma_meta_gap(node);
let gaps: *mut u64 = ma_gaps(node, mt);
unsafe { *gaps.wrapping_offset((offset as i32) as isize) }
}

/// mas_parent_gap() - Set the parent gap and any gaps above, as needed
/// @mas: The maple state
/// @offset: The gap offset in the parent to set
/// @new: The new gap value.
///
/// Set the parent gap then continue to set the gap upwards, using the metadata
/// of the parent to see if it is necessary to check the node above.
fn mas_parent_gap(mas: *mut ma_state, mut offset: u8, mut new: u64) {
let mut pmt: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(pmt) as *mut u32;
let mut meta_offset: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u8 = core::ptr::addr_of_mut!(meta_offset) as *mut u8;
let mut pgaps: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut u64 = core::ptr::addr_of_mut!(pgaps) as *mut *mut u64;
let mut penode: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut maple_enode = core::ptr::addr_of_mut!(penode) as *mut *mut maple_enode;
let mut pnode: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut maple_node = core::ptr::addr_of_mut!(pnode) as *mut *mut maple_node;
let mut meta_gap: u64 = unsafe { core::mem::zeroed() };
unsafe {
meta_gap = 0;
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_4 = mte_parent((*mas).node as *const maple_enode);
*__slate_slot_0 = mas_parent_type(mas, (*mas).node);
*__slate_slot_3 = mt_mk_node(*__slate_slot_4 as *const maple_node, *__slate_slot_0);
*__slate_slot_2 = ma_gaps(*__slate_slot_4, *__slate_slot_0);
'__join_13: {
loop {
if unlikely!((*__slate_slot_0 as u32) != 3) != 0 {
break '__join_13;
} else {
if true {
*__slate_slot_1 = ma_meta_gap(*__slate_slot_4);
meta_gap = *(*__slate_slot_2).wrapping_offset((*__slate_slot_1 as i32) as isize);
*(*__slate_slot_2).wrapping_offset((offset as i32) as isize) = new;
if meta_gap == new {
return;
} else {
if (offset as i32) != (*__slate_slot_1 as i32) {
if meta_gap > new {
return;
} else {
ma_set_meta_gap(*__slate_slot_4, *__slate_slot_0, offset);
}
} else {
if new < meta_gap {
new = ma_max_gap(*__slate_slot_4, *__slate_slot_2, *__slate_slot_0, &mut (*core::ptr::addr_of_mut!(*__slate_slot_1)));
ma_set_meta_gap(*__slate_slot_4, *__slate_slot_0, *__slate_slot_1);
}
}
if ma_is_root(*__slate_slot_4) {
break;
} else {
// Go to the parent node.
*__slate_slot_4 = mte_parent(*__slate_slot_3 as *const maple_enode);
*__slate_slot_0 = mas_parent_type(mas, *__slate_slot_3);
*__slate_slot_2 = ma_gaps(*__slate_slot_4, *__slate_slot_0);
offset = mte_parent_slot(*__slate_slot_3 as *const maple_enode) as u8;
*__slate_slot_3 = mt_mk_node(*__slate_slot_4 as *const maple_node, *__slate_slot_0);
}
}
}
}
}
return;
}
core::arch::asm!("706: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 706b - ., 3; .popsection\n# {0}", const 706 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mas_parent_gap_6, sym __slate_asm_string_mas_parent_gap_7, const 1544 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}

#[inline(always)]
fn mas_update_gap_known(mas: *mut ma_state, gap: u64) {
let pslot: u8 = mte_parent_slot((unsafe { (*mas).node }) as *const maple_enode) as u8;
let p_gap: u64 = unsafe { *ma_gaps(mte_parent((*mas).node as *const maple_enode), mas_parent_type(mas, (*mas).node)).wrapping_offset((pslot as i32) as isize) };
if p_gap != gap {
mas_parent_gap(mas, pslot, gap);
}
}

/// mas_update_gap() - Update a nodes gaps and propagate up if necessary.
/// @mas: the maple state.
fn mas_update_gap(mas: *mut ma_state) {
let mut max_gap: u64 = 0;
if !mt_is_alloc(unsafe { (*mas).tree }) {
return;
}
if mte_is_root((unsafe { (*mas).node }) as *const maple_enode) {
return;
}
max_gap = mas_max_gap(mas);
mas_update_gap_known(mas, max_gap);
}

/// mas_adopt_children() - Set the parent pointer of all nodes in @parent to
/// @parent with the slot encoded.
/// @mas: the maple state (for the tree)
/// @parent: the maple encoded node containing the children.
fn mas_adopt_children(mas: *mut ma_state, parent: *mut maple_enode) {
let r#type: u32 = mte_node_type(parent as *const maple_enode);
let node: *mut maple_node = mte_to_node(parent as *const maple_enode);
let slots: *mut *mut () = ma_slots(node, r#type);
let pivots: *mut u64 = ma_pivots(node, r#type);
let end: u8 = ma_data_end(node, r#type, pivots, unsafe { (*mas).max });
mas_set_parent_slots(mas, parent, slots, 0u8, (end as i32).wrapping_add(1) as u8);
}

// mas_put_in_tree() - Put a new node in the tree, smp_wmb(), and mark the old
// node as dead.
// @mas: the maple state with the new node
// @old_enode: The old maple encoded node to replace.
// @new_height: if we are inserting a root node, update the height of the tree
fn mas_put_in_tree(mas: *mut ma_state, old_enode: *mut maple_enode, new_height: u8) {
let mut offset: u8 = 0;
let mut slots: *mut *mut () = unsafe { core::mem::zeroed() };
if mte_is_root((unsafe { (*mas).node }) as *const maple_enode) {
unsafe {
(*mas_mn(mas as *const ma_state)).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr!(mas_tree_parent!(mas));
}
let _r_a_p__v: u64 = mte_mk_root((unsafe { (*mas).node }) as *const maple_enode) as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_707() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*unsafe { (*mas).tree }).ma_root, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_708() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_709() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { core::ptr::addr_of_mut!((*(*mas).tree).ma_root) }, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
mt_set_height(unsafe { (*mas).tree }, new_height);
} else {
offset = mte_parent_slot((unsafe { (*mas).node }) as *const maple_enode) as u8;
slots = ma_slots(mte_parent((unsafe { (*mas).node }) as *const maple_enode), mas_parent_type(mas, unsafe { (*mas).node }));
let _r_a_p__v: u64 = (unsafe { (*mas).node }) as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_710() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slots.wrapping_offset((offset as i32) as isize), _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_711() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_712() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slots.wrapping_offset((offset as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
mte_set_node_dead(old_enode);
}

/// mas_replace_node() - Replace a node by putting it in the tree, marking it
/// dead, and freeing it.
/// the parent encoding to locate the maple node in the tree.
/// @mas: the ma_state with @mas->node pointing to the new node.
/// @old_enode: The old maple encoded node.
/// @new_height: The new height of the tree as a result of the operation
fn mas_replace_node(mas: *mut ma_state, old_enode: *mut maple_enode, new_height: u8) {
mas_put_in_tree(mas, old_enode, new_height);
mas_free(mas, old_enode);
}

/// mas_find_child() - Find a child who has the parent @mas->node.
/// @mas: the maple state with the parent.
/// @child: the maple state to store the child.
fn mas_find_child(mas: *mut ma_state, child: *mut ma_state) -> bool {
let mt: u32 = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
let mut offset: u8 = 0;
let mut end: u8 = 0;
let node: *mut maple_node = mas_mn(mas as *const ma_state);
let slots: *mut *mut () = ma_slots(node, mt);
let pivots: *mut u64 = ma_pivots(node, mt);
end = ma_data_end(node, mt, pivots, unsafe { (*mas).max });
offset = unsafe { (*mas).offset };
while (offset as i32) <= (end as i32) {
let entry: *mut maple_enode = mas_slot_locked(mas, slots, offset) as *mut maple_enode;
if mte_parent(entry as *const maple_enode) == node {
unsafe {
*child = *mas;
}
unsafe {
(*mas).offset = (offset as i32).wrapping_add(1) as u8;
}
unsafe {
(*child).offset = offset;
}
mas_descend(child);
unsafe {
(*child).offset = 0;
}
return true;
}
offset = (offset as i32).wrapping_add(1) as u8;
}
false
}

/// mas_leaf_set_meta() - Set the metadata of a leaf if possible.
/// @node: The maple node
/// @mt: The maple type
/// @end: The node end
fn mas_leaf_set_meta(node: *mut maple_node, mt: u32, end: u8) {
if (end as i32) < ((unsafe { mt_slots[(mt as u32) as usize] }) as i32).wrapping_sub(1) {
ma_set_meta(node, mt, 0u8, end);
}
}

/// mas_prev_sibling() - Find the previous node with the same parent.
/// @mas: the maple state
///
/// Return: True if there is a previous sibling, false otherwise.
fn mas_prev_sibling(mas: *mut ma_state) -> bool {
let p_slot: u32 = mte_parent_slot((unsafe { (*mas).node }) as *const maple_enode);
// For root node, p_slot is set to 0 by mte_parent_slot().
if p_slot == 0 {
return false;
}
mas_ascend(mas);
unsafe {
(*mas).offset = p_slot.wrapping_sub(1) as u8;
}
mas_descend(mas);
true
}

/// mas_next_sibling() - Find the next node with the same parent.
/// @mas: the maple state
///
/// Return: true if there is a next sibling, false otherwise.
fn mas_next_sibling(mas: *mut ma_state) -> bool {
let mut parent: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = unsafe { (*mas).tree };
__t0.index = unsafe { (*mas).index };
__t0.last = unsafe { (*mas).last };
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
if mte_is_root((unsafe { (*mas).node }) as *const maple_enode) {
return false;
}
parent = unsafe { *mas };
mas_ascend(core::ptr::addr_of_mut!(parent));
parent.offset = mte_parent_slot((unsafe { (*mas).node }) as *const maple_enode).wrapping_add(1) as u8;
if (parent.offset as i32) > (mas_data_end(core::ptr::addr_of_mut!(parent)) as i32) {
return false;
}
unsafe {
*mas = parent;
}
mas_descend(mas);
true
}

/// mas_wr_node_walk() - Find the correct offset for the index in the @mas.
///                      If @mas->index cannot be found within the containing
///                      node, we traverse to the last entry in the node.
/// @wr_mas: The maple write state
///
/// Uses mas_slot_locked() and does not need to worry about dead nodes.
fn mas_wr_node_walk(wr_mas: *mut ma_wr_state) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let mut count: u8 = 0;
let mut offset: u8 = 0;
if unlikely!(ma_is_dense(unsafe { (*wr_mas).r#type })) != 0 {
let __v0: u64 = unsafe { (*mas).index };
unsafe {
(*wr_mas).r_min = __v0;
}
unsafe {
(*wr_mas).r_max = __v0;
}
let __v1: u64 = unsafe { (*mas).min };
unsafe {
(*mas).index = __v1;
}
unsafe {
(*mas).offset = __v1 as u8;
}
return;
}
unsafe {
(*wr_mas).node = mas_mn((*wr_mas).mas as *const ma_state);
}
unsafe {
(*wr_mas).pivots = ma_pivots((*wr_mas).node, (*wr_mas).r#type);
}
let __v2: u8 = ma_data_end(unsafe { (*wr_mas).node }, unsafe { (*wr_mas).r#type }, unsafe { (*wr_mas).pivots }, unsafe { (*mas).max });
unsafe {
(*mas).end = __v2;
}
count = __v2;
offset = unsafe { (*mas).offset };
while (offset as i32) < (count as i32) && (unsafe { (*mas).index }) > unsafe { *unsafe { (*wr_mas).pivots }.wrapping_offset((offset as i32) as isize) } {
offset = (offset as i32).wrapping_add(1) as u8;
}
unsafe {
(*wr_mas).r_max = if (offset as i32) < (count as i32) { *unsafe { (*wr_mas).pivots }.wrapping_offset((offset as i32) as isize) } else { (*mas).max };
}
unsafe {
(*wr_mas).r_min = mas_safe_min(mas, (*wr_mas).pivots, offset);
}
let __v6: u8 = offset;
unsafe {
(*mas).offset = __v6;
}
unsafe {
(*wr_mas).offset_end = __v6;
}
}

fn rebalance_sib(parent: &ma_state, sib: *mut ma_state) {
unsafe {
*sib = *(parent as *const ma_state);
}
// Prioritize move right to pull data left
if ((unsafe { (*sib).offset }) as i32) < ((unsafe { (*sib).end }) as i32) {
unsafe {
(*sib).offset = ((*sib).offset as i32).wrapping_add(1) as u8;
}
} else {
unsafe {
(*sib).offset = ((*sib).offset as i32).wrapping_sub(1) as u8;
}
}
mas_descend(sib);
unsafe {
(*sib).end = mas_data_end(sib);
}
}

fn spanning_sib(l_wr_mas: *mut ma_wr_state, r_wr_mas: *mut ma_wr_state, nneighbour: &mut ma_state) {
let mut l_tmp: ma_state = unsafe { *(*l_wr_mas).mas };
let mut r_tmp: ma_state = unsafe { *(*r_wr_mas).mas };
let mut depth: u8 = 0;
loop {
mas_ascend(core::ptr::addr_of_mut!(r_tmp));
mas_ascend(core::ptr::addr_of_mut!(l_tmp));
depth = (depth as i32).wrapping_add(1) as u8;
if (r_tmp.offset as i32) < (mas_data_end(core::ptr::addr_of_mut!(r_tmp)) as i32) {
r_tmp.offset = (r_tmp.offset as i32).wrapping_add(1) as u8;
mas_descend(core::ptr::addr_of_mut!(r_tmp));
r_tmp.offset = 0;
loop {
let __v5: u8 = (depth as i32).wrapping_sub(1) as u8;
depth = __v5;
if __v5 == 0 {
break;
}
mas_descend(core::ptr::addr_of_mut!(r_tmp));
}
r_tmp.end = mas_data_end(core::ptr::addr_of_mut!(r_tmp));
unsafe {
*(nneighbour as *mut ma_state) = r_tmp;
}
return;
} else {
if l_tmp.offset != 0 {
l_tmp.offset = (l_tmp.offset as i32).wrapping_sub(1) as u8;
loop {
mas_descend(core::ptr::addr_of_mut!(l_tmp));
l_tmp.offset = mas_data_end(core::ptr::addr_of_mut!(l_tmp));
let __v11: u8 = (depth as i32).wrapping_sub(1) as u8;
depth = __v11;
if __v11 == 0 {
break;
}
}
l_tmp.end = l_tmp.offset;
unsafe {
*(nneighbour as *mut ma_state) = l_tmp;
}
return;
}
}
if !(!mte_is_root(r_tmp.node as *const maple_enode)) {
break;
}
}
let mut __v13: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("713: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 713b - ., 3; .popsection\n# {0}", const 713 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_spanning_sib_14, sym __slate_asm_string_spanning_sib_15, const 1844 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v13 = (__ret_warn_on != 0) as i64;
}

/// mas_topiary_node() - Dispose of a single node
/// @mas: The maple state for pushing nodes
/// @in_rcu: If the tree is in rcu mode
///
/// The node will either be RCU freed or pushed back on the maple state.
fn mas_topiary_node(mas: *mut ma_state, tmp_mas: *mut ma_state, in_rcu: bool) {
if mas_is_none(tmp_mas as *const ma_state) {
return;
}
let enode: *mut maple_enode = unsafe { (*tmp_mas).node };
let tmp: *mut maple_node = mte_to_node(enode as *const maple_enode);
mte_set_node_dead(enode);
ma_free_rcu(tmp);
}

/// mas_topiary_replace() - Replace the data with new data, then repair the
/// parent links within the new tree.  Iterate over the dead sub-tree and collect
/// the dead subtrees and topiary the nodes that are no longer of use.
///
/// The new tree will have up to three children with the correct parent.  Keep
/// track of the new entries as they need to be followed to find the next level
/// of new entries.
///
/// The old tree will have up to three children with the old parent.  Keep track
/// of the old entries as they may have more nodes below replaced.  Nodes within
/// [index, last] are dead subtrees, others need to be freed and followed.
///
/// @mas: The maple state pointing at the new data
/// @old_enode: The maple encoded node being replaced
/// @new_height: The new height of the tree as a result of the operation
fn mas_topiary_replace(mas: *mut ma_state, old_enode: *mut maple_enode, new_height: u8) {
let mut tmp: __SlateAlign16<[ma_state; 3]> = __SlateAlign16(unsafe { core::mem::zeroed() });
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp.0) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 264usize);
}
let mut tmp_next: __SlateAlign16<[ma_state; 3]> = __SlateAlign16(unsafe { core::mem::zeroed() });
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp_next.0) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 264usize);
}
let mut subtrees: ma_topiary = ma_topiary { head: core::ptr::null_mut::<maple_enode>(), tail: core::ptr::null_mut::<maple_enode>(), mtree: unsafe { (*mas).tree } };
let mut in_rcu: bool = false;
let mut i: i32 = 0;
let mut n: i32 = 0;
// Place data in tree & then mark node as old
mas_put_in_tree(mas, old_enode, new_height);
// Update the parent pointers in the tree
tmp.0[0] = unsafe { *mas };
tmp.0[0].offset = 0;
tmp.0[1].status = 3;
tmp.0[2].status = 3;
while !mte_is_leaf(tmp.0[0].node as *const maple_enode) {
n = 0;
i = 0;
while i < 3 {
if !mas_is_none((tmp.0.as_mut_ptr() as *mut ma_state).wrapping_offset(i as isize) as *const ma_state) {
while n < 3 {
if !mas_find_child((tmp.0.as_mut_ptr() as *mut ma_state).wrapping_offset(i as isize), (tmp_next.0.as_mut_ptr() as *mut ma_state).wrapping_offset(n as isize)) {
break;
}
n = n.wrapping_add(1);
}
mas_adopt_children((tmp.0.as_mut_ptr() as *mut ma_state).wrapping_offset(i as isize), tmp.0[i as usize].node);
}
i = i.wrapping_add(1);
}
let mut __v6: i64;
let __ret_warn_on: i32 = (n == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("715: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 715b - ., 3; .popsection\n# {0}", const 715 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_topiary_replace_7, sym __slate_asm_string_mas_topiary_replace_8, const 1918 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("716: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 716b - ., 4; .popsection\n# {0}", const 716 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
if __v6 != 0 {
break;
}
while n < 3 {
let __v12: i32 = n;
n = __v12.wrapping_add(1);
tmp_next.0[__v12 as usize].status = 3;
}
i = 0;
while i < 3 {
tmp.0[i as usize] = tmp_next.0[i as usize];
i = i.wrapping_add(1);
}
}
// Collect the old nodes that need to be discarded
if mte_is_leaf(old_enode as *const maple_enode) {
mas_free(mas, old_enode);
return;
}
tmp.0[0] = unsafe { *mas };
tmp.0[0].offset = 0;
tmp.0[0].node = old_enode;
tmp.0[1].status = 3;
tmp.0[2].status = 3;
in_rcu = mt_in_rcu(unsafe { (*mas).tree });
loop {
n = 0;
i = 0;
while i < 3 {
if !mas_is_none((tmp.0.as_mut_ptr() as *mut ma_state).wrapping_offset(i as isize) as *const ma_state) {
while n < 3 {
if !mas_find_child((tmp.0.as_mut_ptr() as *mut ma_state).wrapping_offset(i as isize), (tmp_next.0.as_mut_ptr() as *mut ma_state).wrapping_offset(n as isize)) {
break;
}
if tmp_next.0[n as usize].min >= unsafe { (*(tmp_next.0.as_mut_ptr() as *mut ma_state)).index } && tmp_next.0[n as usize].max <= unsafe { (*(tmp_next.0.as_mut_ptr() as *mut ma_state)).last } {
mat_add(unsafe { &mut (*core::ptr::addr_of_mut!(subtrees)) }, tmp_next.0[n as usize].node);
tmp_next.0[n as usize].status = 3;
} else {
n = n.wrapping_add(1);
}
}
}
i = i.wrapping_add(1);
}
let mut __v24: i64;
let __ret_warn_on: i32 = (n == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("717: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 717b - ., 3; .popsection\n# {0}", const 717 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_topiary_replace_25, sym __slate_asm_string_mas_topiary_replace_26, const 1958 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v24 = (__ret_warn_on != 0) as i64;
if __v24 != 0 {
break;
}
while n < 3 {
let __v30: i32 = n;
n = __v30.wrapping_add(1);
tmp_next.0[__v30 as usize].status = 3;
}
i = 0;
while i < 3 {
mas_topiary_node(mas, (tmp.0.as_mut_ptr() as *mut ma_state).wrapping_offset(i as isize), in_rcu);
tmp.0[i as usize] = tmp_next.0[i as usize];
i = i.wrapping_add(1);
}
if !(!mte_is_leaf(tmp.0[0].node as *const maple_enode)) {
break;
}
}
i = 0;
while i < 3 {
mas_topiary_node(mas, (tmp.0.as_mut_ptr() as *mut ma_state).wrapping_offset(i as isize), in_rcu);
i = i.wrapping_add(1);
}
mas_mat_destroy(mas, unsafe { &mut (*core::ptr::addr_of_mut!(subtrees)) });
}

/// node_copy() - Copy from one node to another.
///
/// @mas: The maple state
/// @src: The source node
/// @start: The offset into the src to start copying
/// @size: The size to copy (non-zero)
/// @s_max: The source node max
/// @s_mt: The source maple node type
/// @dst: The destination
/// @d_start: The start location in the destination node
/// @d_mt: The destination maple node type
fn node_copy(mas: *mut ma_state, src: *mut maple_node, start: u8, mut size: u8, s_max: u64, s_mt: u32, dst: *mut maple_node, d_start: u8, d_mt: u32) -> u64 {
let mut d_gaps: *mut u64 = unsafe { core::mem::zeroed() };
let mut d_max: u64 = 0;
let d_slots: *mut *mut () = ma_slots(dst, d_mt).wrapping_offset((d_start as i32) as isize);
let d_pivots: *mut u64 = ma_pivots(dst, d_mt).wrapping_offset((d_start as i32) as isize);
let s_slots: *mut *mut () = ma_slots(src, s_mt).wrapping_offset((start as i32) as isize);
let s_pivots: *mut u64 = ma_pivots(src, s_mt).wrapping_offset((start as i32) as isize);
unsafe { memcpy(d_slots as *mut (), s_slots as *const (), ((size as i32) as u64).wrapping_mul(core::mem::size_of::<*mut ()>() as u64)) };
if !ma_is_leaf(d_mt) && (s_mt as u32) == 4 {
mas_set_parent_slots(mas, mt_mk_node(dst as *const maple_node, d_mt), d_slots, d_start, size);
}
d_gaps = ma_gaps(dst, d_mt);
if !d_gaps.is_null() {
let s_gaps: *mut u64 = ma_gaps(src, s_mt).wrapping_offset((start as i32) as isize);
let __v0: *mut u64 = d_gaps;
d_gaps = __v0.wrapping_offset((d_start as i32) as isize);
unsafe { memcpy(d_gaps as *mut (), s_gaps as *const (), ((size as i32) as u64).wrapping_mul(core::mem::size_of::<u64>() as u64)) };
}
if (start as i32).wrapping_add(size as i32).wrapping_sub(1) < ((unsafe { mt_pivots[(s_mt as u32) as usize] }) as i32) {
d_max = unsafe { *s_pivots.wrapping_offset((size as i32).wrapping_sub(1) as isize) };
} else {
d_max = s_max;
}
if (d_start as i32).wrapping_add(size as i32) <= ((unsafe { mt_pivots[(d_mt as u32) as usize] }) as i32) {
unsafe {
*d_pivots.wrapping_offset((size as i32).wrapping_sub(1) as isize) = d_max;
}
}
size = (size as i32).wrapping_sub(1) as u8;
if size != 0 {
unsafe { memcpy(d_pivots as *mut (), s_pivots as *const (), ((size as i32) as u64).wrapping_mul(core::mem::size_of::<u64>() as u64)) };
}
d_max
}

/// node_finalise() - Zero out unused area and populate metadata
/// @node: The maple node
/// @mt: The maple node type
/// @end: The end of the used area
fn node_finalise(node: *mut maple_node, mt: u32, end: u8) {
let max_end: u8 = unsafe { mt_slots[(mt as u32) as usize] };
let mut size: u8 = 0;
let mut gap_slot: u8 = 0;
let gaps: *mut u64 = ma_gaps(node, mt);
if (end as i32) < (max_end as i32).wrapping_sub(1) {
size = (max_end as i32).wrapping_sub(end as i32) as u8;
unsafe { memset(ma_slots(node, mt).wrapping_offset((end as i32) as isize) as *mut (), 0i32, ((size as i32) as u64).wrapping_mul(core::mem::size_of::<*mut ()>() as u64)) };
if !gaps.is_null() {
unsafe { memset(gaps.wrapping_offset((end as i32) as isize) as *mut (), 0i32, ((size as i32) as u64).wrapping_mul(core::mem::size_of::<u64>() as u64)) };
}
let __v1: u8 = (size as i32).wrapping_sub(1) as u8;
size = __v1;
if __v1 != 0 {
unsafe { memset(ma_pivots(node, mt).wrapping_offset((end as i32) as isize) as *mut (), 0i32, ((size as i32) as u64).wrapping_mul(core::mem::size_of::<u64>() as u64)) };
}
}
let __v2: bool = !gaps.is_null() && !ma_is_leaf(mt);
if __v2 {
let mut max_gap: u64 = 0;
let mut i: i32 = 0;
while i <= (end as i32) {
if (unsafe { *gaps.wrapping_offset(i as isize) }) > max_gap {
gap_slot = i as u8;
max_gap = unsafe { *gaps.wrapping_offset(i as isize) };
}
i = i.wrapping_add(1);
}
}
if (mt as u32) == 3 {
ma_set_meta(node, mt, gap_slot, (end as i32).wrapping_sub(1) as u8);
} else {
if (end as i32) <= (max_end as i32).wrapping_sub(1) {
ma_set_meta(node, mt, gap_slot, (end as i32).wrapping_sub(1) as u8);
}
}
}

fn mtree_range_walk(mas: *mut ma_state) -> *mut () {
let mut prev_min: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(prev_min) as *mut u64;
let mut prev_max: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u64 = core::ptr::addr_of_mut!(prev_max) as *mut u64;
let mut min: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(min) as *mut u64;
let mut max: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(max) as *mut u64;
let mut end: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u8 = core::ptr::addr_of_mut!(end) as *mut u8;
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut r#type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(r#type) as *mut u32;
let mut last: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut maple_enode = core::ptr::addr_of_mut!(last) as *mut *mut maple_enode;
let mut next: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut maple_enode = core::ptr::addr_of_mut!(next) as *mut *mut maple_enode;
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
let mut offset: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u8 = core::ptr::addr_of_mut!(offset) as *mut u8;
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_8 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_10 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_10 = (*mas).node;
*__slate_slot_4 = (*mas).min;
*__slate_slot_5 = (*mas).max;
loop {
*__slate_slot_9 = *__slate_slot_10;
*__slate_slot_11 = mte_to_node(*__slate_slot_10 as *const maple_enode);
*__slate_slot_8 = mte_node_type(*__slate_slot_10 as *const maple_enode);
*__slate_slot_13 = ma_pivots(*__slate_slot_11, *__slate_slot_8);
*__slate_slot_6 = ma_data_end(*__slate_slot_11, *__slate_slot_8, *__slate_slot_13, *__slate_slot_5);
*__slate_slot_2 = *__slate_slot_4;
*__slate_slot_3 = *__slate_slot_5;
if *(*__slate_slot_13).wrapping_offset(0 as isize) >= (*mas).index {
*__slate_slot_12 = 0;
*__slate_slot_5 = *(*__slate_slot_13).wrapping_offset(0 as isize);
} else {
*__slate_slot_12 = 1;
'__join_4: {
loop {
if (*__slate_slot_12 as i32) < (*__slate_slot_6 as i32) {
if *(*__slate_slot_13).wrapping_offset((*__slate_slot_12 as i32) as isize) >= (*mas).index {
break;
} else {
*__slate_slot_12 = (*__slate_slot_12 as i32).wrapping_add(1) as u8;
}
} else {
break '__join_4;
}
}
*__slate_slot_5 = *(*__slate_slot_13).wrapping_offset((*__slate_slot_12 as i32) as isize);
}
*__slate_slot_4 = unsafe { *(*__slate_slot_13).wrapping_offset((*__slate_slot_12 as i32).wrapping_sub(1) as isize) }.wrapping_add(1);
}
*__slate_slot_7 = ma_slots(*__slate_slot_11, *__slate_slot_8);
*__slate_slot_10 = mt_slot((*mas).tree as *const maple_tree, *__slate_slot_7, *__slate_slot_12) as *mut maple_enode;
if unlikely!(ma_dead_node(*__slate_slot_11 as *const maple_node)) != 0 {
mas_reset(mas);
return core::ptr::null_mut::<()>();
} else {
if ma_is_leaf(*__slate_slot_8) {
break;
}
}
}
(*mas).end = *__slate_slot_6;
(*mas).offset = *__slate_slot_12;
(*mas).index = *__slate_slot_4;
(*mas).last = *__slate_slot_5;
(*mas).min = *__slate_slot_2;
(*mas).max = *__slate_slot_3;
(*mas).node = *__slate_slot_9;
return *__slate_slot_10 as *mut ();
}
}

/// mas_wmb_replace() - Write memory barrier and replace
/// @mas: The maple state
/// @cp: The maple copy node
///
/// Updates gap as necessary.
fn mas_wmb_replace(mas: *mut ma_state, cp: &mut maple_copy) {
let old_enode: *mut maple_enode = unsafe { (*mas).node };
unsafe {
(*mas).node = mt_slot_locked((*mas).tree, (*(cp as *mut maple_copy)).slot.as_mut_ptr() as *mut *mut (), 0u8) as *mut maple_enode;
}
// Insert the new data in the tree
mas_topiary_replace(mas, old_enode, unsafe { (*(cp as *mut maple_copy)).height });
let __v0: bool = mt_is_alloc(unsafe { (*mas).tree }) && !mte_is_root((unsafe { (*mas).node }) as *const maple_enode);
if __v0 {
mas_update_gap_known(mas, unsafe { *((*(cp as *mut maple_copy)).gap.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) });
}
mtree_range_walk(mas);
}

// cp_leaf_init() - Initialize a maple_copy node for the leaf level of a
// spanning store
// @cp: The maple copy node
// @mas: The maple state
// @l_wr_mas: The left write state of the spanning store
// @r_wr_mas: The right write state of the spanning store
fn cp_leaf_init(cp: *mut maple_copy, mas: *mut ma_state, l_wr_mas: *mut ma_wr_state, r_wr_mas: *mut ma_wr_state) {
let mut end: u8 = 0;
// WARNING: The use of RCU_INIT_POINTER() makes it extremely important
// to not expose the maple_copy node to any readers.  Exposure may
// result in buggy code when a compiler reorders the instructions.
unsafe {
(*cp).height = 1;
}
// Create entries to insert including split entries to left and right
if (unsafe { (*l_wr_mas).r_min }) < unsafe { (*mas).index } {
end = (end as i32).wrapping_add(1) as u8;
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_719() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*cp).slot.as_mut_ptr() as *mut *mut () }.wrapping_offset(0 as isize), RCU_INITIALIZER!(unsafe { (*l_wr_mas).content }));
if true {
break;
}
}
unsafe {
*((*cp).__slate_anon_5.pivot.as_mut_ptr() as *mut u64).wrapping_offset(0 as isize) = unsafe { (*mas).index }.wrapping_sub(1);
}
}
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*cp).slot.as_mut_ptr() as *mut *mut () }.wrapping_offset((end as i32) as isize), RCU_INITIALIZER!(unsafe { (*l_wr_mas).entry }));
if true {
break;
}
}
unsafe {
*((*cp).__slate_anon_5.pivot.as_mut_ptr() as *mut u64).wrapping_offset((end as i32) as isize) = (*mas).last;
}
if (unsafe { (*r_wr_mas).end_piv }) > unsafe { (*mas).last } {
end = (end as i32).wrapping_add(1) as u8;
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_721() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { (*cp).slot.as_mut_ptr() as *mut *mut () }.wrapping_offset((end as i32) as isize), RCU_INITIALIZER!(unsafe { *unsafe { (*r_wr_mas).slots }.wrapping_offset(((*r_wr_mas).offset_end as i32) as isize) }));
if true {
break;
}
}
unsafe {
*((*cp).__slate_anon_5.pivot.as_mut_ptr() as *mut u64).wrapping_offset((end as i32) as isize) = (*r_wr_mas).end_piv;
}
}
unsafe {
(*cp).min = (*l_wr_mas).r_min;
}
unsafe {
(*cp).__slate_anon_5.__slate_anon_1.max = *((*cp).__slate_anon_5.pivot.as_mut_ptr() as *mut u64).wrapping_offset((end as i32) as isize);
}
unsafe {
(*cp).end = end;
}
}

/// cp_data_calc() - Calculate the size of the data (1 indexed).
/// @cp: The maple copy struct with the new data populated.
/// @l_wr_mas: The maple write state containing the data to the left of the write
/// @r_wr_mas: The maple write state containing the data to the right of the
/// write
///
/// cp->data is a size (not indexed by 0).
fn cp_data_calc(cp: *mut maple_copy, l_wr_mas: *mut ma_wr_state, r_wr_mas: *mut ma_wr_state) {
// Add 1 every time for the 0th element
unsafe {
(*cp).data = (*(*l_wr_mas).mas).offset;
}
// Add the new data and any partial overwrites
unsafe {
(*cp).data = ((*cp).data as i32).wrapping_add(((*cp).end as i32).wrapping_add(1)) as u8;
}
// Data from right (offset + 1 to end), +1 for zero
unsafe {
(*cp).data = ((*cp).data as i32).wrapping_add(((*(*r_wr_mas).mas).end as i32).wrapping_sub((*r_wr_mas).offset_end as i32)) as u8;
}
}

fn data_fits(sib: *mut ma_state, mas: *mut ma_state, cp: *mut maple_copy) -> bool {
let mut new_data: u8 = 0;
let r#type: u32 = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
let space: u8 = 2i32.wrapping_mul((unsafe { mt_slots[(r#type as u32) as usize] }) as i32) as u8;
let end: u8 = unsafe { (*sib).end };
new_data = (end as i32).wrapping_add(1).wrapping_add((unsafe { (*cp).data }) as i32) as u8;
if (new_data as i32) > (space as i32) {
return false;
}
// This is off by one by design.  The extra space is left to reduce
// jitter in operations that add then remove two entries.
//
// end is an index while new space and data are both sizes.  Adding one
// to end to convert the index to a size means that the below
// calculation should be <=, but we want to keep an extra space in nodes
// to reduce jitter.
//
// Note that it is still possible to get a full node on the left by the
// NULL landing exactly on the split.  The NULL ending of a node happens
// in the dst_setup() function, where we will either increase the split
// by one or decrease it by one, if possible.  In the case of split
// (this case), it is always possible to shift the spilt by one - again
// because there is at least one slot free by the below checking.
if (new_data as i32) < (space as i32) {
return true;
}
false
}

fn push_data_sib(cp: *mut maple_copy, mas: *mut ma_state, sib: *mut ma_state, parent: &ma_state) {
unsafe {
if !mte_is_root((*mas).node as *const maple_enode) {
*sib = *(parent as *const ma_state);
if (*sib).offset != 0 {
(*sib).offset = ((*sib).offset as i32).wrapping_sub(1) as u8;
mas_descend(sib);
(*sib).end = mas_data_end(sib);
if data_fits(sib, mas, cp) {
return;
} else {
// Push left
*sib = *(parent as *const ma_state);
}
}
if ((*sib).offset as i32) < ((*sib).end as i32) {
(*sib).offset = ((*sib).offset as i32).wrapping_add(1) as u8;
mas_descend(sib);
(*sib).end = mas_data_end(sib);
if data_fits(sib, mas, cp) {
return;
}
// Push right
}
}
(*sib).end = 0;
}
}

/// rebalance_data() - Calculate the @cp data, populate @sib if insufficient or
/// if the data can be pushed into a sibling.
/// @cp: The maple copy node
/// @wr_mas: The left write maple state
/// @sib: The maple state of the sibling.
///
/// Note: @cp->data is a size and not indexed by 0. @sib->end may be set to 0 to
/// indicate it will not be used.
fn rebalance_data(cp: *mut maple_copy, wr_mas: *mut ma_wr_state, sib: *mut ma_state, parent: &ma_state) {
unsafe {
'__join_0: {
cp_data_calc(cp, wr_mas, wr_mas);
(*sib).end = 0;
if ((*cp).data as i32) > (mt_slots[((*wr_mas).r#type as u32) as usize] as i32) {
push_data_sib(cp, (*wr_mas).mas, sib, parent);
if (*sib).end != 0 {
break '__join_0;
}
} else {
if ((*cp).data as i32) <= (mt_min_slots[((*wr_mas).r#type as u32) as usize] as i32) {
if (*(*wr_mas).mas).min != 0 || (*(*wr_mas).mas).max != !0 {
rebalance_sib(parent, sib);
break '__join_0;
}
}
}
return;
}
(*cp).data = ((*cp).data as i32).wrapping_add(((*sib).end as i32).wrapping_add(1)) as u8;
}
}

/// spanning_data() - Calculate the @cp data and populate @sib if insufficient
/// @cp: The maple copy node
/// @l_wr_mas: The left write maple state
/// @r_wr_mas: The right write maple state
/// @sib: The maple state of the sibling.
///
/// Note: @cp->data is a size and not indexed by 0. @sib->end may be set to 0 to
/// indicate it will not be used.
fn spanning_data(cp: *mut maple_copy, l_wr_mas: *mut ma_wr_state, r_wr_mas: *mut ma_wr_state, sib: &mut ma_state) {
cp_data_calc(cp, l_wr_mas, r_wr_mas);
if ((unsafe { (*(*l_wr_mas).mas).min }) != 0 || (unsafe { (*(*r_wr_mas).mas).max }) != !0) && ((unsafe { (*cp).data }) as i32) <= ((unsafe { mt_min_slots[((*l_wr_mas).r#type as u32) as usize] }) as i32) {
spanning_sib(l_wr_mas, r_wr_mas, sib);
unsafe {
(*cp).data = ((*cp).data as i32).wrapping_add(((*(sib as *mut ma_state)).end as i32).wrapping_add(1)) as u8;
}
} else {
unsafe {
(*(sib as *mut ma_state)).end = 0;
}
}
}

/// dst_setup() - Set up one or more destinations for the new data.
/// @cp: The maple copy node
/// @mas: The maple state
/// @mt: The source node type
fn dst_setup(cp: *mut maple_copy, mas: *mut ma_state, mt: u32) {
let mut i: i32 = unsafe { core::mem::zeroed() };
let mut s_off: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u8 = core::ptr::addr_of_mut!(s_off) as *mut u8;
let mut s: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u8 = core::ptr::addr_of_mut!(s) as *mut u8;
let mut off: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u8 = core::ptr::addr_of_mut!(off) as *mut u8;
unsafe {
'__join_3: {
// Data is 1 indexed, every src has +1 added.
if ((*cp).data as i32) <= (mt_slots[(mt as u32) as usize] as i32) {
(*cp).split = ((*cp).data as i32).wrapping_sub(1) as u8;
(*cp).d_count = 1;
} else {
(*cp).split = (((*cp).data as i32).wrapping_sub(1) / 2) as u8;
(*cp).d_count = 2;
if ((*cp).data as i32) >= (mt_slots[(mt as u32) as usize] as i32).wrapping_mul(2) {
if ((*cp).data as i32) == (mt_slots[(mt as u32) as usize] as i32).wrapping_mul(2) {
let __slate_auto_bytes_1 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
if !ma_is_leaf(mt) {
break '__join_3;
} else {
// Leaf nodes are a bit tricky because we cannot assume the data
// can fit due to the NULL limitation on node ends.
*__slate_slot_19 = (*cp).split;
*__slate_slot_18 = 0;
while (*__slate_slot_18 as i32) < ((*cp).s_count as i32) {
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_17 = ((*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((*__slate_slot_18 as i32) as isize)).end as i32).wrapping_sub((*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((*__slate_slot_18 as i32) as isize)).start as i32) as u8;
if (*__slate_slot_17 as i32) >= (*__slate_slot_19 as i32) {
break;
} else {
*__slate_slot_17 = (*__slate_slot_17 as i32).wrapping_add(1) as u8;
*__slate_slot_19 = (*__slate_slot_19 as i32).wrapping_sub(*__slate_slot_17 as i32) as u8;
*__slate_slot_18 = (*__slate_slot_18 as i32).wrapping_add(1) as u8;
}
}
let __v10: u8 = *__slate_slot_19;
let __v9: u8 = (__v10 as i32).wrapping_add((*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((*__slate_slot_18 as i32) as isize)).start as i32) as u8;
*__slate_slot_19 = __v9;
if !unsafe { *ma_slots((*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((*__slate_slot_18 as i32) as isize)).node, (*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((*__slate_slot_18 as i32) as isize)).mt).wrapping_offset((*__slate_slot_19 as i32) as isize) }.is_null() {
break '__join_3;
} else {
(*cp).split = ((*cp).split as i32).wrapping_add(1) as u8;
if ((*cp).split as i32) < (mt_slots[(mt as u32) as usize] as i32) {
break '__join_3;
} else {
(*cp).split = ((*cp).split as i32).wrapping_sub(2) as u8;
if ((*cp).data as i32).wrapping_sub(2).wrapping_sub((*cp).split as i32) < (mt_slots[(mt as u32) as usize] as i32) {
break '__join_3;
}
}
}
}
}
// No other choice but to 3-way split the data
(*cp).split = (((*cp).data as i32).wrapping_add(2) / 3) as u8;
(*cp).d_count = 3;
}
}
}
i = 0;
while i < ((*cp).d_count as i32) {
(*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(i as isize)).mt = mt;
(*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(i as isize)).node = ma_mnode_ptr!(mas_pop_node(mas));
i = i.wrapping_add(1);
}
}
}

fn append_mas_cp(cp: *mut maple_copy, mas: *mut ma_state, start: u8, end: u8) {
let mut mt: u32 = 0;
let count: u8 = unsafe { (*cp).s_count };
let node: *mut maple_node = mas_mn(mas as *const ma_state);
mt = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).node = node;
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).mt = mt;
}
if ((unsafe { (*mas).end }) as i32) <= (end as i32) {
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).max = (*mas).max;
}
} else {
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).max = *ma_pivots(node, mt).wrapping_offset((end as i32) as isize);
}
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).start = start;
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).end = end;
}
unsafe {
(*cp).s_count = ((*cp).s_count as i32).wrapping_add(1) as u8;
}
}

fn append_wr_mas_cp(cp: *mut maple_copy, wr_mas: *mut ma_wr_state, start: u8, end: u8) {
let count: u8 = unsafe { (*cp).s_count };
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).node = (*wr_mas).node;
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).mt = (*wr_mas).r#type;
}
if ((unsafe { (*(*wr_mas).mas).end }) as i32) <= (end as i32) {
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).max = (*(*wr_mas).mas).max;
}
} else {
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).max = *unsafe { (*wr_mas).pivots }.wrapping_offset((end as i32) as isize);
}
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).start = start;
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((count as i32) as isize)).end = end;
}
unsafe {
(*cp).s_count = ((*cp).s_count as i32).wrapping_add(1) as u8;
}
}

fn init_cp_src(cp: *mut maple_copy) {
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset(((*cp).s_count as i32) as isize)).node = ma_mnode_ptr__2!(cp);
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset(((*cp).s_count as i32) as isize)).mt = 4;
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset(((*cp).s_count as i32) as isize)).max = (*cp).__slate_anon_5.__slate_anon_1.max;
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset(((*cp).s_count as i32) as isize)).start = 0;
}
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset(((*cp).s_count as i32) as isize)).end = (*cp).end;
}
unsafe {
(*cp).s_count = ((*cp).s_count as i32).wrapping_add(1) as u8;
}
}

/// multi_src_setup() - Set the @cp node up with multiple sources to copy from.
/// @cp: The maple copy node
/// @l_wr_mas: The left write maple state
/// @r_wr_mas: The right write maple state
/// @sib: The sibling maple state
///
/// Note: @sib->end == 0 indicates no sibling will be used.
fn multi_src_setup(cp: *mut maple_copy, l_wr_mas: *mut ma_wr_state, r_wr_mas: *mut ma_wr_state, sib: *mut ma_state) {
unsafe {
(*cp).s_count = 0;
}
if (unsafe { (*sib).end }) != 0 && (unsafe { (*sib).max }) < unsafe { (*(*l_wr_mas).mas).min } {
append_mas_cp(cp, sib, 0u8, unsafe { (*sib).end });
}
// Copy left 0 - offset
if (unsafe { (*(*l_wr_mas).mas).offset }) != 0 {
let off: u8 = ((unsafe { (*(*l_wr_mas).mas).offset }) as i32).wrapping_sub(1) as u8;
append_wr_mas_cp(cp, l_wr_mas, 0u8, off);
unsafe {
(*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset(((*cp).s_count as i32).wrapping_sub(1) as isize)).max = unsafe { (*cp).min }.wrapping_sub(1);
}
}
init_cp_src(cp);
// Copy right either from offset or offset + 1 pending on r_max
if ((unsafe { (*(*r_wr_mas).mas).end }) as i32) != ((unsafe { (*r_wr_mas).offset_end }) as i32) {
append_wr_mas_cp(cp, r_wr_mas, ((unsafe { (*r_wr_mas).offset_end }) as i32).wrapping_add(1) as u8, unsafe { (*(*r_wr_mas).mas).end });
}
if (unsafe { (*sib).end }) != 0 && (unsafe { (*sib).min }) > unsafe { (*(*r_wr_mas).mas).max } {
append_mas_cp(cp, sib, 0u8, unsafe { (*sib).end });
}
}

fn cp_data_write(cp: &mut maple_copy, mas: *mut ma_state) {
let mut dst: *mut maple_node = unsafe { core::mem::zeroed() };
let mut src: *mut maple_node = unsafe { core::mem::zeroed() };
let mut s: u8 = 0;
let mut d: u8 = 0;
let mut dst_offset: u8 = 0;
let mut data_offset: u8 = 0;
let mut src_end: u8 = 0;
let mut s_offset: u8 = 0;
let mut split: u8 = 0;
let mut s_max: u64 = 0;
let mut d_max: u64 = 0;
let mut dst_size: u8 = 0;
let mut s_mt: u32 = 0;
let mut d_mt: u32 = 0;
// Readability help
src = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).node };
dst = unsafe { (*((*(cp as *mut maple_copy)).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).node };
s_offset = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).start };
src_end = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).end };
split = unsafe { (*(cp as *mut maple_copy)).split };
s_max = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).max };
s_mt = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).mt };
d_mt = unsafe { (*((*(cp as *mut maple_copy)).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).mt };
loop {
dst_offset = 0;
d_max = 0;
dst = unsafe { (*((*(cp as *mut maple_copy)).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).node };
d_mt = unsafe { (*((*(cp as *mut maple_copy)).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).mt };
dst_size = (split as i32).wrapping_add(1) as u8;
while dst_size != 0 {
let size: u8 = if (src_end as i32).wrapping_sub(s_offset as i32).wrapping_add(1) < (dst_size as i32) {
(src_end as i32).wrapping_sub(s_offset as i32).wrapping_add(1) as u8
} else {
dst_size
};
d_max = node_copy(mas, src, s_offset, size, s_max, s_mt, dst, dst_offset, d_mt);
dst_offset = (dst_offset as i32).wrapping_add(size as i32) as u8;
s_offset = (s_offset as i32).wrapping_add(size as i32) as u8;
if (s_offset as i32) > (src_end as i32) {
// This source is exhausted
s = (s as i32).wrapping_add(1) as u8;
if (s as i32) >= ((unsafe { (*(cp as *mut maple_copy)).s_count }) as i32) {
unsafe {
(*((*(cp as *mut maple_copy)).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).max = d_max;
}
node_finalise(dst, d_mt, dst_offset);
return;
}
// Reset local src
src = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).node };
s_offset = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).start };
src_end = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).end };
s_max = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).max };
s_mt = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).mt };
}
dst_size = (dst_size as i32).wrapping_sub(size as i32) as u8;
data_offset = (data_offset as i32).wrapping_add(size as i32) as u8;
}
split = unsafe { (*(cp as *mut maple_copy)).split };
unsafe {
(*((*(cp as *mut maple_copy)).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).max = d_max;
}
// Handle null entries
let __v11: bool = (unsafe { (*((*(cp as *mut maple_copy)).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).max }) != !0 && unsafe { *ma_slots(dst, d_mt).wrapping_offset((dst_offset as i32).wrapping_sub(1) as isize) }.is_null();
if __v11 {
if (s_offset as i32) == ((unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).start }) as i32) {
s = (s as i32).wrapping_sub(1) as u8;
src = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).node };
src_end = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).end };
s_max = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).max };
s_mt = unsafe { (*((*(cp as *mut maple_copy)).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset((s as i32) as isize)).mt };
s_offset = src_end;
} else {
s_offset = (s_offset as i32).wrapping_sub(1) as u8;
}
// Set dst max and clear pivot
split = (split as i32).wrapping_add(1) as u8;
data_offset = (data_offset as i32).wrapping_sub(1) as u8;
dst_offset = (dst_offset as i32).wrapping_sub(1) as u8;
unsafe {
(*((*(cp as *mut maple_copy)).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).max = *ma_pivots(dst, d_mt).wrapping_offset((dst_offset as i32).wrapping_sub(1) as isize);
}
}
node_finalise(dst, d_mt, dst_offset);
d = (d as i32).wrapping_add(1) as u8; // Next destination
if (d as i32) == ((unsafe { (*(cp as *mut maple_copy)).d_count }) as i32).wrapping_sub(1) {
split = ((unsafe { (*(cp as *mut maple_copy)).data }) as i32).wrapping_sub(data_offset as i32) as u8;
}
if (d as i32) >= ((unsafe { (*(cp as *mut maple_copy)).d_count }) as i32) {
let mut __v24: i64;
let __ret_warn_on: i32 = ((data_offset as i32) < ((unsafe { (*(cp as *mut maple_copy)).data }) as i32)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("722: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 722b - ., 3; .popsection\n# {0}", const 722 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cp_data_write_25, sym __slate_asm_string_cp_data_write_26, const 2600 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("723: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 723b - ., 4; .popsection\n# {0}", const 723 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v24 = (__ret_warn_on != 0) as i64;
return;
}
if (data_offset as i32) > ((unsafe { (*(cp as *mut maple_copy)).data }) as i32) {
break;
}
}
}

/// cp_dst_to_slots() - Migrate the maple copy destination to the maple copy
/// slots
/// @cp: The maple copy node
/// @min: The minimal value represented
/// @max: The maximum value represented
/// @mas: The maple state
fn cp_dst_to_slots(cp: *mut maple_copy, min: u64, max: u64, mas: *mut ma_state) {
let mut d: u8 = 0;
let mut slot_min: u64 = min;
while (d as i32) < ((unsafe { (*cp).d_count }) as i32) {
let mn: *mut maple_node = unsafe { (*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).node };
let mt: u32 = unsafe { (*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).mt };
let slot_max: u64 = unsafe { (*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset((d as i32) as isize)).max };
// Warning, see cp_leaf_init() comment and rcu_assign_pointer()
// documentation.  Since these are new nodes, there are no
// read-side operations that can view them until they are
// inserted into the tree after an rcu_assign_pointer() call.
ma_init_slot(unsafe { (*cp).slot.as_mut_ptr() as *mut *mut () }.wrapping_offset((d as i32) as isize), mn as *const maple_node, mt);
unsafe {
*((*cp).__slate_anon_5.pivot.as_mut_ptr() as *mut u64).wrapping_offset((d as i32) as isize) = slot_max;
}
if mt_is_alloc(unsafe { (*mas).tree }) {
if ma_is_leaf(mt) {
unsafe {
*((*cp).gap.as_mut_ptr() as *mut u64).wrapping_offset((d as i32) as isize) = ma_leaf_max_gap(mn, mt, slot_min, slot_max, ma_pivots(mn, mt), ma_slots(mn, mt));
}
} else {
let gaps: *mut u64 = ma_gaps(mn, mt);
if !gaps.is_null() {
let gap_slot: u8 = ma_meta_gap(mn);
unsafe {
*((*cp).gap.as_mut_ptr() as *mut u64).wrapping_offset((d as i32) as isize) = *gaps.wrapping_offset((gap_slot as i32) as isize);
}
}
}
}
slot_min = slot_max.wrapping_add(1);
d = (d as i32).wrapping_add(1) as u8;
}
unsafe {
(*cp).end = ((*cp).d_count as i32).wrapping_sub(1) as u8;
}
unsafe {
(*cp).min = min;
}
unsafe {
(*cp).__slate_anon_5.__slate_anon_1.max = max;
}
}

fn cp_is_new_root(cp: *mut maple_copy, mas: *mut ma_state) -> bool {
if (unsafe { (*cp).min }) != 0 || (unsafe { (*cp).__slate_anon_5.__slate_anon_1.max }) != !0 {
return false;
}
if ((unsafe { (*cp).d_count }) as i32) != 1 {
let mut mt: u32 = 3;
if !mt_is_alloc(unsafe { (*mas).tree }) {
mt = 2;
}
unsafe {
(*cp).data = (*cp).d_count;
}
unsafe {
(*cp).s_count = 0;
}
dst_setup(cp, mas, mt);
init_cp_src(cp);
node_copy(mas, unsafe { (*((*cp).src.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_167).wrapping_offset(0 as isize)).node }, 0u8, unsafe { (*cp).data }, unsafe { (*cp).__slate_anon_5.__slate_anon_1.max }, 4u32, unsafe { (*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(0 as isize)).node }, 0u8, mt);
node_finalise(unsafe { (*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(0 as isize)).node }, mt, ((unsafe { (*cp).end }) as i32).wrapping_add(1) as u8);
// Warning, see cp_leaf_init() comment and rcu_assign_pointer()
// documentation.  Since this is a new root, there are no
// read-side operations that can view it until it is insert into
// the tree after an rcu_assign_pointer() call.
ma_init_slot(unsafe { (*cp).slot.as_mut_ptr() as *mut *mut () }.wrapping_offset(0 as isize), (unsafe { (*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(0 as isize)).node }) as *const maple_node, mt);
unsafe {
(*cp).height = ((*cp).height as i32).wrapping_add(1) as u8;
}
}
let mut __v3: i64;
let __ret_warn_on: i32 = ((unsafe { (*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(0 as isize)).node }) != mte_to_node(mt_slot_locked(unsafe { (*mas).tree }, unsafe { (*cp).slot.as_mut_ptr() as *mut *mut () }, 0u8) as *const maple_enode)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("724: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 724b - ., 3; .popsection\n# {0}", const 724 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_cp_is_new_root_4, sym __slate_asm_string_cp_is_new_root_5, const 2686 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("725: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 725b - ., 4; .popsection\n# {0}", const 725 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
unsafe {
(*(*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(0 as isize)).node).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr!(mas_tree_parent!(mas));
}
unsafe {
(*mas).min = 0;
}
unsafe {
(*mas).max = !0;
}
unsafe {
(*mas).depth = 0;
}
unsafe {
(*mas).node = mas_root_locked(mas) as *mut maple_enode;
}
true
}

fn cp_converged(cp: *mut maple_copy, mas: *mut ma_state, sib: &ma_state) -> bool {
if ((unsafe { (*cp).d_count }) as i32) != 1 || (unsafe { (*(sib as *const ma_state)).end }) != 0 {
return false;
}
unsafe {
(*(*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(0 as isize)).node).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr__2!(unsafe { (*mas_mn(mas as *const ma_state)).__slate_anon_0.__slate_anon_0.parent });
}
true
}

/// spanning_ascend() - See if a spanning store operation has to keep walking up
/// the tree
/// @cp: The maple_copy node
/// @l_wr_mas: The left maple write state
/// @r_wr_mas: The right maple write state
/// @sib: the maple state of the sibling
///
/// Returns: True if another iteration is necessary.
fn spanning_ascend(cp: *mut maple_copy, mas: *mut ma_state, l_wr_mas: *mut ma_wr_state, r_wr_mas: *mut ma_wr_state, sib: &ma_state) -> bool {
if (unsafe { (*(sib as *const ma_state)).end }) != 0 {
if (unsafe { (*(sib as *const ma_state)).max }) < unsafe { (*(*l_wr_mas).mas).min } {
unsafe {
*(*l_wr_mas).mas = *(sib as *const ma_state);
}
} else {
unsafe {
*(*r_wr_mas).mas = *(sib as *const ma_state);
}
}
}
cp_dst_to_slots(cp, unsafe { (*(*l_wr_mas).mas).min }, unsafe { (*(*r_wr_mas).mas).max }, mas);
if cp_is_new_root(cp, mas) {
return false;
}
// Converged and has a single destination
if ((unsafe { (*cp).d_count }) as i32) == 1 && (unsafe { (*(*l_wr_mas).mas).node }) == unsafe { (*(*r_wr_mas).mas).node } {
unsafe {
(*(*((*cp).dst.as_mut_ptr() as *mut __SlateRecord_maple_tree_h_162).wrapping_offset(0 as isize)).node).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr__2!(unsafe { (*mas_mn(mas as *const ma_state)).__slate_anon_0.__slate_anon_0.parent });
}
return false;
}
unsafe {
(*cp).height = ((*cp).height as i32).wrapping_add(1) as u8;
}
wr_mas_ascend(l_wr_mas);
wr_mas_ascend(r_wr_mas);
true
}

fn copy_tree_location(src: *const ma_state, dst: *mut ma_state) {
unsafe {
(*dst).node = (*src).node;
}
unsafe {
(*dst).offset = (*src).offset;
}
unsafe {
(*dst).min = (*src).min;
}
unsafe {
(*dst).max = (*src).max;
}
unsafe {
(*dst).end = (*src).end;
}
unsafe {
(*dst).depth = (*src).depth;
}
}

/// rebalance_ascend() - Ascend the tree and set up for the next loop - if
/// necessary
///
/// Return: True if there another rebalancing operation on the next level is
/// needed, false otherwise.
fn rebalance_ascend(cp: *mut maple_copy, wr_mas: *mut ma_wr_state, sib: &ma_state, parent: &mut ma_state) -> bool {
let mut min: u64 = 0;
let mut max: u64 = 0;
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
if (unsafe { (*(sib as *const ma_state)).end }) == 0 {
min = unsafe { (*mas).min };
max = unsafe { (*mas).max };
} else {
if (unsafe { (*(sib as *const ma_state)).min }) > unsafe { (*mas).max } {
// Move right succeeded
min = unsafe { (*mas).min };
max = unsafe { (*(sib as *const ma_state)).max };
unsafe {
(*wr_mas).offset_end = ((*(parent as *mut ma_state)).offset as i32).wrapping_add(1) as u8;
}
} else {
min = unsafe { (*(sib as *const ma_state)).min };
max = unsafe { (*mas).max };
unsafe {
(*wr_mas).offset_end = (*(parent as *mut ma_state)).offset;
}
unsafe {
(*parent).offset = ((*parent).offset as i32).wrapping_sub(1) as u8;
}
}
}
cp_dst_to_slots(cp, min, max, mas);
if cp_is_new_root(cp, mas) {
return false;
}
if cp_converged(cp, mas, sib) {
return false;
}
unsafe {
(*cp).height = ((*cp).height as i32).wrapping_add(1) as u8;
}
copy_tree_location((parent as *mut ma_state) as *const ma_state, mas);
wr_mas_setup(wr_mas, mas);
true
}

// mas_root_expand() - Expand a root to a node
// @mas: The maple state
// @entry: The entry to store into the tree
fn mas_root_expand(mas: *mut ma_state, entry: *mut ()) {
let contents: *mut () = mas_root_locked(mas);
let r#type: u32 = 1;
let mut slots: *mut *mut () = unsafe { core::mem::zeroed() };
let mut slot: i32 = 0;
let node: *mut maple_node = mas_pop_node(mas);
let pivots: *mut u64 = ma_pivots(node, r#type);
slots = ma_slots(node, r#type);
unsafe {
(*node).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr!(mas_tree_parent!(mas));
}
unsafe {
(*mas).node = mt_mk_node(node as *const maple_node, r#type);
}
unsafe {
(*mas).status = 0;
}
if (unsafe { (*mas).index }) != 0 {
if !contents.is_null() {
let _r_a_p__v: u64 = contents as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_726() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slots.wrapping_offset(slot as isize), _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_727() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_728() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slots.wrapping_offset(slot as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
if likely!((unsafe { (*mas).index }) > 1) != 0 {
slot = slot.wrapping_add(1);
}
}
let __v12: i32 = slot;
slot = __v12.wrapping_add(1);
unsafe {
*pivots.wrapping_offset(__v12 as isize) = unsafe { (*mas).index }.wrapping_sub(1);
}
}
{
let _r_a_p__v: u64 = entry as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_729() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slots.wrapping_offset(slot as isize), _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_730() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_731() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slots.wrapping_offset(slot as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
(*mas).offset = slot as u8;
}
unsafe {
*pivots.wrapping_offset(slot as isize) = (*mas).last;
}
if (unsafe { (*mas).last }) != !0 {
let __v25: i32 = slot.wrapping_add(1);
slot = __v25;
unsafe {
*pivots.wrapping_offset(__v25 as isize) = !0;
}
}
mt_set_height(unsafe { (*mas).tree }, 1u8);
ma_set_meta(node, 1u32, 0u8, slot as u8);
// swap the new root into the tree
let _r_a_p__v: u64 = mte_mk_root((unsafe { (*mas).node }) as *const maple_enode) as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_732() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*unsafe { (*mas).tree }).ma_root, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_733() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_734() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { core::ptr::addr_of_mut!((*(*mas).tree).ma_root) }, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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

// mas_store_root() - Storing value into root.
// @mas: The maple state
// @entry: The entry to store.
//
// There is no root node now and we are storing a value into the root - this
// function either assigns the pointer or expands into a node.
fn mas_store_root(mas: *mut ma_state, entry: *mut ()) {
if entry.is_null() {
if (unsafe { (*mas).index }) == 0 {
let _r_a_p__v: u64 = 0;
if true && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_735() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*unsafe { (*mas).tree }).ma_root, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_736() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_737() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { core::ptr::addr_of_mut!((*(*mas).tree).ma_root) }, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
} else {
if likely!((unsafe { (*mas).last }) != 0 || (unsafe { (*mas).index }) != 0) != 0 {
mas_root_expand(mas, entry);
} else {
if (entry as u64) & 3 == 2 {
mas_root_expand(mas, entry);
} else {
let _r_a_p__v: u64 = entry as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_738() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!((*unsafe { (*mas).tree }).ma_root, _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_739() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_740() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*unsafe { core::ptr::addr_of_mut!((*(*mas).tree).ma_root) }, RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
(*mas).status = 1;
}
}
}
}
}

/// mas_is_span_wr() - Check if the write needs to be treated as a write that
/// spans the node.
/// @wr_mas: The maple write state
///
/// Spanning writes are writes that start in one node and end in another OR if
/// the write of a %NULL will cause the node to end with a %NULL.
///
/// Return: True if this is a spanning write, false otherwise.
fn mas_is_span_wr(wr_mas: *mut ma_wr_state) -> bool {
let mut max: u64 = unsafe { (*wr_mas).r_max };
let last: u64 = unsafe { (*(*wr_mas).mas).last };
let r#type: u32 = unsafe { (*wr_mas).r#type };
let entry: *mut () = unsafe { (*wr_mas).entry };
// Contained in this pivot, fast path
if last < max {
return false;
}
if ma_is_leaf(r#type) {
max = unsafe { (*(*wr_mas).mas).max };
if last < max {
return false;
}
}
if last == max {
// The last entry of leaf node cannot be NULL unless it is the
// rightmost node (writing ULONG_MAX), otherwise it spans slots.
if !entry.is_null() || last == !0 {
return false;
}
}
let mut __v0: *const u8;
__v0 = unsafe { ___tp_str__2897 };
trace_ma_write(__v0, unsafe { (*wr_mas).mas }, unsafe { (*wr_mas).r_max }, entry);
true
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__2897: *const u8 = (b"mas_is_span_wr\0".as_ptr() as *mut u8) as *const u8;

fn mas_wr_walk_descend(wr_mas: *mut ma_wr_state) {
unsafe {
(*wr_mas).r#type = mte_node_type((*(*wr_mas).mas).node as *const maple_enode);
}
mas_wr_node_walk(wr_mas);
unsafe {
(*wr_mas).slots = ma_slots((*wr_mas).node, (*wr_mas).r#type);
}
}

fn mas_wr_walk_traverse(wr_mas: *mut ma_wr_state) {
unsafe {
(*(*wr_mas).mas).max = (*wr_mas).r_max;
}
unsafe {
(*(*wr_mas).mas).min = (*wr_mas).r_min;
}
unsafe {
(*(*wr_mas).mas).node = (*wr_mas).content as *mut maple_enode;
}
unsafe {
(*(*wr_mas).mas).offset = 0;
}
let __v0: *mut ma_state = unsafe { (*wr_mas).mas };
unsafe {
(*__v0).depth = ((*__v0).depth as i32).wrapping_add(1) as u8;
}
}

/// mas_wr_walk() - Walk the tree for a write.
/// @wr_mas: The maple write state
///
/// Uses mas_slot_locked() and does not need to worry about dead nodes.
///
/// Return: True if it's contained in a node, false on spanning write.
fn mas_wr_walk(wr_mas: *mut ma_wr_state) -> bool {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
while true {
mas_wr_walk_descend(wr_mas);
if unlikely!(mas_is_span_wr(wr_mas)) != 0 {
return false;
}
unsafe {
(*wr_mas).content = mas_slot_locked(mas, (*wr_mas).slots, (*mas).offset);
}
if ma_is_leaf(unsafe { (*wr_mas).r#type }) {
return true;
}
if ((unsafe { (*mas).end }) as i32) < ((unsafe { mt_slots[((*wr_mas).r#type as u32) as usize] }) as i32).wrapping_sub(1) {
unsafe {
(*wr_mas).vacant_height = ((*mas).depth as i32).wrapping_add(1) as u8;
}
}
if ma_is_root(mas_mn(mas as *const ma_state)) {
// root needs more than 2 entries to be sufficient + 1
if ((unsafe { (*mas).end }) as i32) > 2 {
unsafe {
(*wr_mas).sufficient_height = 1;
}
}
} else {
if ((unsafe { (*mas).end }) as i32) > ((unsafe { mt_min_slots[((*wr_mas).r#type as u32) as usize] }) as i32).wrapping_add(1) {
unsafe {
(*wr_mas).sufficient_height = ((*mas).depth as i32).wrapping_add(1) as u8;
}
}
}
mas_wr_walk_traverse(wr_mas);
}
true
}

fn mas_wr_walk_index(wr_mas: *mut ma_wr_state) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
while true {
mas_wr_walk_descend(wr_mas);
unsafe {
(*wr_mas).content = mas_slot_locked(mas, (*wr_mas).slots, (*mas).offset);
}
if ma_is_leaf(unsafe { (*wr_mas).r#type }) {
return;
}
mas_wr_walk_traverse(wr_mas);
}
}

/// mas_extend_spanning_null() - Extend a store of a %NULL to include surrounding %NULLs.
/// @l_wr_mas: The left maple write state
/// @r_wr_mas: The right maple write state
fn mas_extend_spanning_null(l_wr_mas: *mut ma_wr_state, r_wr_mas: &ma_wr_state) {
let r_mas: *mut ma_state = unsafe { (*(r_wr_mas as *const ma_wr_state)).mas };
let l_mas: *mut ma_state = unsafe { (*l_wr_mas).mas };
let l_slot: u8 = unsafe { (*l_mas).offset };
if unsafe { (*l_wr_mas).content }.is_null() {
unsafe {
(*l_mas).index = (*l_wr_mas).r_min;
}
}
let __v0: bool = (unsafe { (*l_mas).index }) == unsafe { (*l_wr_mas).r_min } && (l_slot != 0 && mas_slot_locked(l_mas, unsafe { (*l_wr_mas).slots }, (l_slot as i32).wrapping_sub(1) as u8).is_null());
if __v0 {
if (l_slot as i32) > 1 {
unsafe {
(*l_mas).index = unsafe { *unsafe { (*l_wr_mas).pivots }.wrapping_offset((l_slot as i32).wrapping_sub(2) as isize) }.wrapping_add(1);
}
} else {
unsafe {
(*l_mas).index = (*l_mas).min;
}
}
unsafe {
(*l_mas).offset = (l_slot as i32).wrapping_sub(1) as u8;
}
unsafe {
(*l_wr_mas).r_min = (*l_mas).index;
}
}
if unsafe { (*(r_wr_mas as *const ma_wr_state)).content }.is_null() {
if (unsafe { (*r_mas).last }) < unsafe { (*(r_wr_mas as *const ma_wr_state)).r_max } {
unsafe {
(*r_mas).last = (*(r_wr_mas as *const ma_wr_state)).r_max;
}
}
unsafe {
(*r_mas).offset = ((*r_mas).offset as i32).wrapping_add(1) as u8;
}
}
}

fn mas_state_walk(mas: *mut ma_state) -> *mut () {
let entry: *mut () = mas_start(mas) as *mut ();
if mas_is_none(mas as *const ma_state) {
return core::ptr::null_mut::<()>();
}
if mas_is_ptr(mas as *const ma_state) {
return entry;
}
mtree_range_walk(mas)
}

/// mtree_lookup_walk() - Internal quick lookup that does not keep maple state up
/// to date.
///
/// @mas: The maple state.
///
/// Note: Leaves mas in undesirable state.
/// Return: The entry for @mas->index or %NULL on dead node.
fn mtree_lookup_walk(mas: *mut ma_state) -> *mut () {
let mut end: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u8 = core::ptr::addr_of_mut!(end) as *mut u8;
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut r#type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(r#type) as *mut u32;
let mut next: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut maple_enode = core::ptr::addr_of_mut!(next) as *mut *mut maple_enode;
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
let mut offset: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u8 = core::ptr::addr_of_mut!(offset) as *mut u8;
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_5 = (*mas).node;
loop {
*__slate_slot_6 = mte_to_node(*__slate_slot_5 as *const maple_enode);
*__slate_slot_4 = mte_node_type(*__slate_slot_5 as *const maple_enode);
*__slate_slot_8 = ma_pivots(*__slate_slot_6, *__slate_slot_4);
*__slate_slot_2 = mt_pivots[(*__slate_slot_4 as u32) as usize];
*__slate_slot_7 = 0;
loop {
if *(*__slate_slot_8).wrapping_offset((*__slate_slot_7 as i32) as isize) >= (*mas).index {
break;
} else {
let __v0: u8 = (*__slate_slot_7 as i32).wrapping_add(1) as u8;
*__slate_slot_7 = __v0;
if (__v0 as i32) >= (*__slate_slot_2 as i32) {
break;
}
}
}
*__slate_slot_3 = ma_slots(*__slate_slot_6, *__slate_slot_4);
*__slate_slot_5 = mt_slot((*mas).tree as *const maple_tree, *__slate_slot_3, *__slate_slot_7) as *mut maple_enode;
if unlikely!(ma_dead_node(*__slate_slot_6 as *const maple_node)) != 0 {
mas_reset(mas);
return core::ptr::null_mut::<()>();
} else {
if ma_is_leaf(*__slate_slot_4) {
break;
}
}
}
return *__slate_slot_5 as *mut ();
}
}

// mas_new_root() - Create a new root node that only contains the entry passed
// in.
// @mas: The maple state
// @entry: The entry to store.
//
// Only valid when the index == 0 and the last == ULONG_MAX
fn mas_new_root(mas: *mut ma_state, entry: *mut ()) {
let mut _r_a_p__v: u64 = unsafe { core::mem::zeroed() };
let mut _r_a_p__v_1: u64 = unsafe { core::mem::zeroed() };
let mut _r_a_p__v_2: u64 = unsafe { core::mem::zeroed() };
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
let mut r#type: u32 = unsafe { core::mem::zeroed() };
let mut root: *mut maple_enode = unsafe { core::mem::zeroed() };
unsafe {
root = mas_root_locked(mas) as *mut maple_enode;
r#type = 1;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __ret_warn_on: i32 = ((*mas).index != 0 || (*mas).last != !0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("741: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 741b - ., 3; .popsection\n# {0}", const 741 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_new_root_17, sym __slate_asm_string_mas_new_root_18, const 3077 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("742: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 742b - ., 4; .popsection\n# {0}", const 742 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if entry.is_null() {
mt_set_height((*mas).tree, 0u8);
_r_a_p__v_2 = entry as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE!(*core::ptr::addr_of_mut!((*(*mas).tree).ma_root), RCU_INITIALIZER!(_r_a_p__v_2 as *mut ()));
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
(*mas).status = 1;
} else {
*__slate_slot_8 = mas_pop_node(mas);
*__slate_slot_6 = ma_pivots(*__slate_slot_8, r#type);
*__slate_slot_7 = ma_slots(*__slate_slot_8, r#type);
(*(*__slate_slot_8)).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr!(mas_tree_parent!(mas));
(*mas).node = mt_mk_node(*__slate_slot_8 as *const maple_node, r#type);
(*mas).status = 0;
_r_a_p__v_1 = entry as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE!(*(*__slate_slot_7).wrapping_offset(0 as isize), RCU_INITIALIZER!(_r_a_p__v_1 as *mut ()));
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
*(*__slate_slot_6).wrapping_offset(0 as isize) = (*mas).last;
mt_set_height((*mas).tree, 1u8);
_r_a_p__v = mte_mk_root((*mas).node as *const maple_enode) as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE!(*core::ptr::addr_of_mut!((*(*mas).tree).ma_root), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
if xa_is_node(root as *const ()) {
mte_destroy_walk(root, (*mas).tree);
}
}
}

/// mas_wr_spanning_store() - Create a subtree with the store operation completed
/// and new nodes where necessary, then place the sub-tree in the actual tree.
/// Note that mas is expected to point to the node which caused the store to
/// span.
/// @wr_mas: The maple write state
fn mas_wr_spanning_store(wr_mas: *mut ma_wr_state) {
let mut cp: maple_copy = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 256usize);
}
let mut sib: ma_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sib) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 88usize);
}
// Left and Right side of spanning store
let mut r_mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = core::ptr::null_mut::<maple_tree>();
__t0.index = 0u64;
__t0.last = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
let mut r_wr_mas: ma_wr_state = {
let mut __t1: ma_wr_state = unsafe { core::mem::zeroed() };
__t1.mas = core::ptr::addr_of_mut!(r_mas);
__t1.entry = unsafe { (*wr_mas).entry };
__t1.content = core::ptr::null_mut::<()>();
__t1.vacant_height = 0u8;
__t1.sufficient_height = 0u8;
__t1
};
// A store operation that spans multiple nodes is called a spanning
// store and is handled early in the store call stack by the function
// mas_is_span_wr().  When a spanning store is identified, the maple
// state is duplicated.  The first maple state walks the left tree path
// to ``index``, the duplicate walks the right tree path to ``last``.
// The data in the two nodes are combined into a single node, two nodes,
// or possibly three nodes (see the 3-way split above).  A ``NULL``
// written to the last entry of a node is considered a spanning store as
// a rebalance is required for the operation to complete and an overflow
// of data may happen.
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let mut __v0: *const u8;
__v0 = unsafe { ___tp_str__3131 };
trace_ma_op(__v0, mas);
if unlikely!((unsafe { (*mas).index }) == 0 && (unsafe { (*mas).last }) == !0) != 0 {
mas_new_root(mas, unsafe { (*wr_mas).entry });
return;
}
// Node rebalancing may occur due to this store, so there may be three new
// entries per level plus a new root.
// Set up right side.  Need to get to the next offset after the spanning
// store to ensure it's not NULL and to combine both the next node and
// the node with the start together.
r_mas = unsafe { *mas };
// Avoid overflow, walk to next slot in the tree.
if r_mas.last.wrapping_add(1) != 0 {
let __v1: u64 = r_mas.last;
r_mas.last = __v1.wrapping_add(1);
}
r_mas.index = r_mas.last;
mas_wr_walk_index(core::ptr::addr_of_mut!(r_wr_mas));
let __v3: u64 = unsafe { (*mas).last };
r_mas.index = __v3;
r_mas.last = __v3;
r_wr_mas.end_piv = r_wr_mas.r_max;
// Set up left side.
mas_wr_walk_index(wr_mas);
if unsafe { (*wr_mas).entry }.is_null() {
mas_extend_spanning_null(wr_mas, unsafe { &(*core::ptr::addr_of_mut!(r_wr_mas)) });
unsafe {
(*mas).last = r_mas.last;
}
}
// expanding NULLs may make this cover the entire range
if (unsafe { (*mas).index }) == 0 && r_mas.last == !0 {
mas_set_range(mas, 0u64, 18446744073709551615u64);
mas_new_root(mas, unsafe { (*wr_mas).entry });
return;
}
cp_leaf_init(core::ptr::addr_of_mut!(cp), mas, wr_mas, core::ptr::addr_of_mut!(r_wr_mas));
loop {
spanning_data(core::ptr::addr_of_mut!(cp), wr_mas, core::ptr::addr_of_mut!(r_wr_mas), unsafe { &mut (*core::ptr::addr_of_mut!(sib)) });
multi_src_setup(core::ptr::addr_of_mut!(cp), wr_mas, core::ptr::addr_of_mut!(r_wr_mas), core::ptr::addr_of_mut!(sib));
dst_setup(core::ptr::addr_of_mut!(cp), mas, unsafe { (*wr_mas).r#type });
cp_data_write(unsafe { &mut (*core::ptr::addr_of_mut!(cp)) }, mas);
if !spanning_ascend(core::ptr::addr_of_mut!(cp), mas, wr_mas, core::ptr::addr_of_mut!(r_wr_mas), unsafe { &(*core::ptr::addr_of_mut!(sib)) }) {
break;
}
}
mas_wmb_replace(mas, unsafe { &mut (*core::ptr::addr_of_mut!(cp)) });
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__3131: *const u8 = (b"mas_wr_spanning_store\0".as_ptr() as *mut u8) as *const u8;

// mas_wr_node_store() - Attempt to store the value in a node
// @wr_mas: The maple write state
//
// Attempts to reuse the node, but may allocate.
fn mas_wr_node_store(wr_mas: *mut ma_wr_state) {
let mut clear_from: u8 = unsafe { core::mem::zeroed() };
let mut _r_a_p__v: u64 = unsafe { core::mem::zeroed() };
let mut _r_a_p__v_1: u64 = unsafe { core::mem::zeroed() };
let mut in_rcu: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut bool = core::ptr::addr_of_mut!(in_rcu) as *mut bool;
let mut mas: core::mem::MaybeUninit<*mut ma_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut ma_state = core::ptr::addr_of_mut!(mas) as *mut *mut ma_state;
let mut new_end: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u8 = core::ptr::addr_of_mut!(new_end) as *mut u8;
let mut dst_slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut *mut () = core::ptr::addr_of_mut!(dst_slots) as *mut *mut *mut ();
let mut dst_pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut u64 = core::ptr::addr_of_mut!(dst_pivots) as *mut *mut u64;
let mut newnode: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut maple_node = core::ptr::addr_of_mut!(newnode) as *mut *mut maple_node;
let mut reuse: core::mem::MaybeUninit<maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut maple_node = core::ptr::addr_of_mut!(reuse) as *mut maple_node;
let mut node_slots: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u8 = core::ptr::addr_of_mut!(node_slots) as *mut u8;
let mut node_pivots: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u8 = core::ptr::addr_of_mut!(node_pivots) as *mut u8;
let mut copy_size: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u8 = core::ptr::addr_of_mut!(copy_size) as *mut u8;
let mut offset_end: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut u8 = core::ptr::addr_of_mut!(offset_end) as *mut u8;
let mut dst_offset: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut u8 = core::ptr::addr_of_mut!(dst_offset) as *mut u8;
unsafe {
let __slate_auto_bytes = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_4 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_5 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 256usize);
let __slate_auto_bytes_6 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_9 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_10 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_16 = (*wr_mas).mas;
trace_ma_op(___tp_str__3198, *__slate_slot_16);
*__slate_slot_15 = mt_in_rcu((*(*__slate_slot_16)).tree);
*__slate_slot_25 = (*wr_mas).offset_end;
*__slate_slot_23 = mt_pivots[((*wr_mas).r#type as u32) as usize];
*__slate_slot_22 = mt_slots[((*wr_mas).r#type as u32) as usize];
// Assume last adds an entry
*__slate_slot_17 = ((*(*__slate_slot_16)).end as i32).wrapping_add(1).wrapping_sub(*__slate_slot_25 as i32).wrapping_add((*(*__slate_slot_16)).offset as i32) as u8;
if (*(*__slate_slot_16)).last == (*wr_mas).end_piv {
*__slate_slot_25 = (*__slate_slot_25 as i32).wrapping_add(1) as u8; // don't copy this offset
*__slate_slot_17 = (*__slate_slot_17 as i32).wrapping_sub(1) as u8;
}
// set up node.
if *__slate_slot_15 {
*__slate_slot_20 = mas_pop_node(*__slate_slot_16);
} else {
*__slate_slot_20 = core::ptr::addr_of_mut!(*__slate_slot_21);
}
(*(*__slate_slot_20)).__slate_anon_0.__slate_anon_0.parent = (*mas_mn(*__slate_slot_16 as *const ma_state)).__slate_anon_0.__slate_anon_0.parent;
*__slate_slot_19 = ma_pivots(*__slate_slot_20, (*wr_mas).r#type);
*__slate_slot_18 = ma_slots(*__slate_slot_20, (*wr_mas).r#type);
// Copy from start to insert point
if (*(*__slate_slot_16)).offset != 0 {
memcpy(*__slate_slot_19 as *mut (), (*wr_mas).pivots as *const (), (core::mem::size_of::<u64>() as u64).wrapping_mul(((*(*__slate_slot_16)).offset as i32) as u64));
memcpy(*__slate_slot_18 as *mut (), (*wr_mas).slots as *const (), (core::mem::size_of::<*mut ()>() as u64).wrapping_mul(((*(*__slate_slot_16)).offset as i32) as u64));
}
// Handle insert of new range starting after old range
if (*wr_mas).r_min < (*(*__slate_slot_16)).index {
_r_a_p__v_1 = (*wr_mas).content as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE!(*(*__slate_slot_18).wrapping_offset(((*(*__slate_slot_16)).offset as i32) as isize), RCU_INITIALIZER!(_r_a_p__v_1 as *mut ()));
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
let __v8: *mut ma_state = *__slate_slot_16;
let __v7: u8 = (*__v8).offset;
(*__v8).offset = (__v7 as i32).wrapping_add(1) as u8;
*(*__slate_slot_19).wrapping_offset((__v7 as i32) as isize) = unsafe { (*(*__slate_slot_16)).index }.wrapping_sub(1);
*__slate_slot_17 = (*__slate_slot_17 as i32).wrapping_add(1) as u8;
}
// Store the new entry and range end.
if ((*(*__slate_slot_16)).offset as i32) < (*__slate_slot_23 as i32) {
*(*__slate_slot_19).wrapping_offset(((*(*__slate_slot_16)).offset as i32) as isize) = (*(*__slate_slot_16)).last;
}
_r_a_p__v = (*wr_mas).entry as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE!(*(*__slate_slot_18).wrapping_offset(((*(*__slate_slot_16)).offset as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
// this range wrote to the end of the node or it overwrote the rest of
// the data
if (*__slate_slot_25 as i32) <= ((*(*__slate_slot_16)).end as i32) {
*__slate_slot_26 = ((*(*__slate_slot_16)).offset as i32).wrapping_add(1) as u8;
// Copy to the end of node if necessary.
*__slate_slot_24 = ((*(*__slate_slot_16)).end as i32).wrapping_sub(*__slate_slot_25 as i32).wrapping_add(1) as u8;
memcpy((*__slate_slot_18).wrapping_offset((*__slate_slot_26 as i32) as isize) as *mut (), unsafe { (*wr_mas).slots }.wrapping_offset((*__slate_slot_25 as i32) as isize) as *const (), (core::mem::size_of::<*mut ()>() as u64).wrapping_mul((*__slate_slot_24 as i32) as u64));
memcpy((*__slate_slot_19).wrapping_offset((*__slate_slot_26 as i32) as isize) as *mut (), unsafe { (*wr_mas).pivots }.wrapping_offset((*__slate_slot_25 as i32) as isize) as *const (), (core::mem::size_of::<u64>() as u64).wrapping_mul((*__slate_slot_24 as i32).wrapping_sub(1) as u64));
if (*__slate_slot_17 as i32) < (*__slate_slot_23 as i32) {
*(*__slate_slot_19).wrapping_offset((*__slate_slot_17 as i32) as isize) = (*(*__slate_slot_16)).max;
}
}
if !(*__slate_slot_15) && (*__slate_slot_17 as i32).wrapping_add(2) < (*__slate_slot_22 as i32) {
clear_from = (*__slate_slot_17 as i32).wrapping_add(1) as u8;
// Note that the last slot is never cleared, since the metadata
// will be stored there or it has a value.
memset((*__slate_slot_18).wrapping_offset((clear_from as i32) as isize) as *mut (), 0i32, (core::mem::size_of::<*mut ()>() as u64).wrapping_mul((*__slate_slot_22 as i32).wrapping_sub(clear_from as i32) as u64));
if (clear_from as i32) < (*__slate_slot_23 as i32) {
memset((*__slate_slot_19).wrapping_offset((clear_from as i32) as isize) as *mut (), 0i32, (core::mem::size_of::<u64>() as u64).wrapping_mul((*__slate_slot_23 as i32).wrapping_sub(clear_from as i32) as u64));
}
}
mas_leaf_set_meta(*__slate_slot_20, (*wr_mas).r#type, *__slate_slot_17);
if *__slate_slot_15 {
let old_enode: *mut maple_enode = (*(*__slate_slot_16)).node;
(*(*__slate_slot_16)).node = mt_mk_node(*__slate_slot_20 as *const maple_node, (*wr_mas).r#type);
mas_replace_node(*__slate_slot_16, old_enode, mas_mt_height(*__slate_slot_16) as u8);
} else {
memcpy((*wr_mas).node as *mut (), *__slate_slot_20 as *const (), core::mem::size_of::<maple_node>() as u64);
}
trace_ma_write(___tp_str__3280, *__slate_slot_16, 0u64, (*wr_mas).entry);
mas_update_gap(*__slate_slot_16);
(*(*__slate_slot_16)).end = *__slate_slot_17;
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__3198: *const u8 = (b"mas_wr_node_store\0".as_ptr() as *mut u8) as *const u8;

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__3280: *const u8 = (b"mas_wr_node_store\0".as_ptr() as *mut u8) as *const u8;

// mas_wr_slot_store: Attempt to store a value in a slot.
// @wr_mas: the maple write state
fn mas_wr_slot_store(wr_mas: *mut ma_wr_state) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let offset: u8 = unsafe { (*mas).offset };
let slots: *mut *mut () = unsafe { (*wr_mas).slots };
let mut gap: bool = false;
let __v0: bool = gap;
gap = (__v0 as i32) | (unsafe { (*wr_mas).content }.is_null() as i32) != 0;
let __v2: i32 = mt_slot_locked(unsafe { (*mas).tree }, slots, (offset as i32).wrapping_add(1) as u8).is_null() as i32;
gap = (gap as i32) | __v2 != 0;
if ((unsafe { (*wr_mas).offset_end }) as i32).wrapping_sub(offset as i32) == 1 {
if (unsafe { (*mas).index }) == unsafe { (*wr_mas).r_min } {
// Overwriting the range and a part of the next one
let _r_a_p__v: u64 = (unsafe { (*wr_mas).entry }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*slots.wrapping_offset((offset as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((offset as i32) as isize) = (*mas).last;
}
} else {
// Overwriting a part of the range and the next one
let _r_a_p__v: u64 = (unsafe { (*wr_mas).entry }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*slots.wrapping_offset((offset as i32).wrapping_add(1) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((offset as i32) as isize) = unsafe { (*mas).index }.wrapping_sub(1);
}
unsafe {
(*mas).offset = ((*mas).offset as i32).wrapping_add(1) as u8;
}
// Keep mas accurate.
}
} else {
let mut __v22: i64;
let __ret_warn_on: i32 = !(!mt_in_rcu(unsafe { (*mas).tree })) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("764: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 764b - ., 3; .popsection\n# {0}", const 764 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_wr_slot_store_23, sym __slate_asm_string_mas_wr_slot_store_24, const 3311 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("765: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 765b - ., 4; .popsection\n# {0}", const 765 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v22 = (__ret_warn_on != 0) as i64;
// Expand the range, only partially overwriting the previous and
// next ranges
let __v27: i32 = mt_slot_locked(unsafe { (*mas).tree }, slots, (offset as i32).wrapping_add(2) as u8).is_null() as i32;
gap = (gap as i32) | __v27 != 0;
let _r_a_p__v: u64 = (unsafe { (*wr_mas).entry }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*slots.wrapping_offset((offset as i32).wrapping_add(1) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((offset as i32) as isize) = unsafe { (*mas).index }.wrapping_sub(1);
}
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((offset as i32).wrapping_add(1) as isize) = (*mas).last;
}
unsafe {
(*mas).offset = ((*mas).offset as i32).wrapping_add(1) as u8;
}
// Keep mas accurate.
}
let mut __v40: *const u8;
__v40 = unsafe { ___tp_str__3323 };
trace_ma_write(__v40, mas, 0u64, unsafe { (*wr_mas).entry });
// Only update gap when the new entry is empty or there is an empty
// entry in the original two ranges.
if unsafe { (*wr_mas).entry }.is_null() || gap {
mas_update_gap(mas);
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__3323: *const u8 = (b"mas_wr_slot_store\0".as_ptr() as *mut u8) as *const u8;

fn mas_wr_extend_null(wr_mas: *mut ma_wr_state) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
if unsafe { *unsafe { (*wr_mas).slots }.wrapping_offset(((*wr_mas).offset_end as i32) as isize) }.is_null() {
// If this one is null, the next and prev are not
unsafe {
(*mas).last = (*wr_mas).end_piv;
}
} else {
// Check next slot(s) if we are overwriting the end
if (unsafe { (*mas).last }) == unsafe { (*wr_mas).end_piv } && ((unsafe { (*mas).end }) as i32) != ((unsafe { (*wr_mas).offset_end }) as i32) && unsafe { *unsafe { (*wr_mas).slots }.wrapping_offset(((*wr_mas).offset_end as i32).wrapping_add(1) as isize) }.is_null() {
unsafe {
(*wr_mas).offset_end = ((*wr_mas).offset_end as i32).wrapping_add(1) as u8;
}
if ((unsafe { (*wr_mas).offset_end }) as i32) == ((unsafe { (*mas).end }) as i32) {
unsafe {
(*mas).last = (*mas).max;
}
} else {
unsafe {
(*mas).last = *unsafe { (*wr_mas).pivots }.wrapping_offset(((*wr_mas).offset_end as i32) as isize);
}
}
unsafe {
(*wr_mas).end_piv = (*mas).last;
}
}
}
if unsafe { (*wr_mas).content }.is_null() {
// If this one is null, the next and prev are not
unsafe {
(*mas).index = (*wr_mas).r_min;
}
} else {
// Check prev slot if we are overwriting the start
if (unsafe { (*mas).index }) == unsafe { (*wr_mas).r_min } && (unsafe { (*mas).offset }) != 0 && unsafe { *unsafe { (*wr_mas).slots }.wrapping_offset(((*mas).offset as i32).wrapping_sub(1) as isize) }.is_null() {
unsafe {
(*mas).offset = ((*mas).offset as i32).wrapping_sub(1) as u8;
}
let __v6: u64 = mas_safe_min(mas, unsafe { (*wr_mas).pivots }, unsafe { (*mas).offset });
unsafe {
(*mas).index = __v6;
}
unsafe {
(*wr_mas).r_min = __v6;
}
unsafe {
(*wr_mas).r_max = *unsafe { (*wr_mas).pivots }.wrapping_offset(((*mas).offset as i32) as isize);
}
}
}
}

fn mas_wr_end_piv(wr_mas: *mut ma_wr_state) {
while ((unsafe { (*wr_mas).offset_end }) as i32) < ((unsafe { (*(*wr_mas).mas).end }) as i32) && (unsafe { (*(*wr_mas).mas).last }) > unsafe { *unsafe { (*wr_mas).pivots }.wrapping_offset(((*wr_mas).offset_end as i32) as isize) } {
unsafe {
(*wr_mas).offset_end = ((*wr_mas).offset_end as i32).wrapping_add(1) as u8;
}
}
if ((unsafe { (*wr_mas).offset_end }) as i32) < ((unsafe { (*(*wr_mas).mas).end }) as i32) {
unsafe {
(*wr_mas).end_piv = *unsafe { (*wr_mas).pivots }.wrapping_offset(((*wr_mas).offset_end as i32) as isize);
}
} else {
unsafe {
(*wr_mas).end_piv = (*(*wr_mas).mas).max;
}
}
}

fn mas_wr_new_end(wr_mas: *mut ma_wr_state) -> u8 {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let mut new_end: u8 = ((unsafe { (*mas).end }) as i32).wrapping_add(2) as u8;
new_end = (new_end as i32).wrapping_sub(((unsafe { (*wr_mas).offset_end }) as i32).wrapping_sub((unsafe { (*mas).offset }) as i32)) as u8;
if (unsafe { (*wr_mas).r_min }) == unsafe { (*mas).index } {
new_end = (new_end as i32).wrapping_sub(1) as u8;
}
if (unsafe { (*wr_mas).end_piv }) == unsafe { (*mas).last } {
new_end = (new_end as i32).wrapping_sub(1) as u8;
}
new_end
}

// mas_wr_append: Attempt to append
// @wr_mas: the maple write state
//
// This is currently unsafe in rcu mode since the end of the node may be cached
// by readers while the node contents may be updated which could result in
// inaccurate information.
fn mas_wr_append(wr_mas: *mut ma_wr_state) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let mut slots: *mut *mut () = unsafe { core::mem::zeroed() };
let end: u8 = unsafe { (*mas).end };
let new_end: u8 = mas_wr_new_end(wr_mas);
if (new_end as i32) < ((unsafe { mt_pivots[((*wr_mas).r#type as u32) as usize] }) as i32) {
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((new_end as i32) as isize) = *unsafe { (*wr_mas).pivots }.wrapping_offset((end as i32) as isize);
}
ma_set_meta(unsafe { (*wr_mas).node }, unsafe { (*wr_mas).r#type }, 0u8, new_end);
}
slots = unsafe { (*wr_mas).slots };
if (new_end as i32) == (end as i32).wrapping_add(1) {
if (unsafe { (*mas).last }) == unsafe { (*wr_mas).r_max } {
// Append to end of range
let _r_a_p__v: u64 = (unsafe { (*wr_mas).entry }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*slots.wrapping_offset((new_end as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((end as i32) as isize) = unsafe { (*mas).index }.wrapping_sub(1);
}
unsafe {
(*mas).offset = new_end;
}
} else {
// Append to start of range
{
let _r_a_p__v: u64 = (unsafe { (*wr_mas).content }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*slots.wrapping_offset((new_end as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((end as i32) as isize) = (*mas).last;
}
let _r_a_p__v: u64 = (unsafe { (*wr_mas).entry }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*slots.wrapping_offset((end as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
} else {
// Append to the range without touching any boundaries.
{
let _r_a_p__v: u64 = (unsafe { (*wr_mas).content }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*slots.wrapping_offset((new_end as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((end as i32).wrapping_add(1) as isize) = (*mas).last;
}
let _r_a_p__v: u64 = (unsafe { (*wr_mas).entry }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*slots.wrapping_offset((end as i32).wrapping_add(1) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
unsafe {
*unsafe { (*wr_mas).pivots }.wrapping_offset((end as i32) as isize) = unsafe { (*mas).index }.wrapping_sub(1);
}
unsafe {
(*mas).offset = (end as i32).wrapping_add(1) as u8;
}
}
if unsafe { (*wr_mas).content }.is_null() || unsafe { (*wr_mas).entry }.is_null() {
mas_update_gap(mas);
}
unsafe {
(*mas).end = new_end;
}
let mut __v35: *const u8;
__v35 = unsafe { ___tp_str__3441 };
trace_ma_write(__v35, mas, new_end as u64, unsafe { (*wr_mas).entry });
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__3441: *const u8 = (b"mas_wr_append\0".as_ptr() as *mut u8) as *const u8;

/// split_ascend() - See if a split operation has to keep walking up the tree
/// @cp: The maple_copy node
/// @wr_mas: The maple write state
/// @sib: the maple state of the sibling
///
/// Return: true if another split operation on the next level is needed, false
/// otherwise
fn split_ascend(cp: *mut maple_copy, wr_mas: *mut ma_wr_state, sib: &ma_state, parent: &mut ma_state) -> bool {
let mut min: u64 = 0;
let mut max: u64 = 0;
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
min = unsafe { (*mas).min }; // push right, or normal split
max = unsafe { (*mas).max };
unsafe {
(*wr_mas).offset_end = (*(parent as *mut ma_state)).offset;
}
if (unsafe { (*(sib as *const ma_state)).end }) != 0 {
if (unsafe { (*(sib as *const ma_state)).max }) < unsafe { (*mas).min } {
min = unsafe { (*(sib as *const ma_state)).min }; // push left
unsafe {
(*parent).offset = ((*parent).offset as i32).wrapping_sub(1) as u8;
}
} else {
max = unsafe { (*(sib as *const ma_state)).max }; // push right
unsafe {
(*wr_mas).offset_end = ((*wr_mas).offset_end as i32).wrapping_add(1) as u8;
}
}
}
cp_dst_to_slots(cp, min, max, mas);
if cp_is_new_root(cp, mas) {
return false;
}
if cp_converged(cp, mas, sib) {
return false;
}
unsafe {
(*cp).height = ((*cp).height as i32).wrapping_add(1) as u8;
}
copy_tree_location((parent as *mut ma_state) as *const ma_state, mas);
wr_mas_setup(wr_mas, mas);
true
}

/// split_data() - Calculate the @cp data, populate @sib if the data can be
/// pushed into a sibling.
/// @cp: The maple copy node
/// @wr_mas: The left write maple state
/// @sib: The maple state of the sibling.
///
/// Note: @cp->data is a size and not indexed by 0. @sib->end may be set to 0 to
/// indicate it will not be used.
fn split_data(cp: *mut maple_copy, wr_mas: *mut ma_wr_state, sib: *mut ma_state, parent: &ma_state) {
cp_data_calc(cp, wr_mas, wr_mas);
if ((unsafe { (*cp).data }) as i32) <= ((unsafe { mt_slots[((*wr_mas).r#type as u32) as usize] }) as i32) {
unsafe {
(*sib).end = 0;
}
return;
}
push_data_sib(cp, unsafe { (*wr_mas).mas }, sib, parent);
if (unsafe { (*sib).end }) != 0 {
unsafe {
(*cp).data = ((*cp).data as i32).wrapping_add(((*sib).end as i32).wrapping_add(1)) as u8;
}
}
}

/// mas_wr_split() - Expand one node into two
/// @wr_mas: The write maple state
fn mas_wr_split(wr_mas: *mut ma_wr_state) {
let mut parent: ma_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(parent) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 88usize);
}
let mut cp: maple_copy = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 256usize);
}
let mut sib: ma_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sib) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 88usize);
}
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let mut __v0: *const u8;
__v0 = unsafe { ___tp_str__3525 };
trace_ma_write(__v0, unsafe { (*wr_mas).mas }, 0u64, unsafe { (*wr_mas).entry });
parent = unsafe { *mas };
cp_leaf_init(core::ptr::addr_of_mut!(cp), mas, wr_mas, wr_mas);
loop {
if !mte_is_root(parent.node as *const maple_enode) {
mas_ascend(core::ptr::addr_of_mut!(parent));
parent.end = mas_data_end(core::ptr::addr_of_mut!(parent));
}
split_data(core::ptr::addr_of_mut!(cp), wr_mas, core::ptr::addr_of_mut!(sib), unsafe { &(*core::ptr::addr_of_mut!(parent)) });
multi_src_setup(core::ptr::addr_of_mut!(cp), wr_mas, wr_mas, core::ptr::addr_of_mut!(sib));
dst_setup(core::ptr::addr_of_mut!(cp), mas, unsafe { (*wr_mas).r#type });
cp_data_write(unsafe { &mut (*core::ptr::addr_of_mut!(cp)) }, mas);
if !split_ascend(core::ptr::addr_of_mut!(cp), wr_mas, unsafe { &(*core::ptr::addr_of_mut!(sib)) }, unsafe { &mut (*core::ptr::addr_of_mut!(parent)) }) {
break;
}
}
mas_wmb_replace(mas, unsafe { &mut (*core::ptr::addr_of_mut!(cp)) });
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__3525: *const u8 = (b"mas_wr_split\0".as_ptr() as *mut u8) as *const u8;

/// mas_wr_rebalance() - Insufficient data in one node needs to either get data
/// from a sibling or absorb a sibling all together.
/// @wr_mas: The write maple state
///
/// Rebalance is different than a spanning store in that the write state is
/// already at the leaf node that's being altered.
fn mas_wr_rebalance(wr_mas: *mut ma_wr_state) {
let mut parent: ma_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(parent) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 88usize);
}
let mut cp: maple_copy = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(cp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 256usize);
}
let mut sib: ma_state = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sib) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 88usize);
}
// Rebalancing occurs if a node is insufficient.  Data is rebalanced
// against the node to the right if it exists, otherwise the node to the
// left of this node is rebalanced against this node.  If rebalancing
// causes just one node to be produced instead of two, then the parent
// is also examined and rebalanced if it is insufficient.  Every level
// tries to combine the data in the same way.  If one node contains the
// entire range of the tree, then that node is used as a new root node.
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let mut __v0: *const u8;
__v0 = unsafe { ___tp_str__3568 };
trace_ma_op(__v0, mas);
parent = unsafe { *mas };
cp_leaf_init(core::ptr::addr_of_mut!(cp), mas, wr_mas, wr_mas);
loop {
if !mte_is_root(parent.node as *const maple_enode) {
mas_ascend(core::ptr::addr_of_mut!(parent));
parent.end = mas_data_end(core::ptr::addr_of_mut!(parent));
}
rebalance_data(core::ptr::addr_of_mut!(cp), wr_mas, core::ptr::addr_of_mut!(sib), unsafe { &(*core::ptr::addr_of_mut!(parent)) });
multi_src_setup(core::ptr::addr_of_mut!(cp), wr_mas, wr_mas, core::ptr::addr_of_mut!(sib));
dst_setup(core::ptr::addr_of_mut!(cp), mas, unsafe { (*wr_mas).r#type });
cp_data_write(unsafe { &mut (*core::ptr::addr_of_mut!(cp)) }, mas);
if !rebalance_ascend(core::ptr::addr_of_mut!(cp), wr_mas, unsafe { &(*core::ptr::addr_of_mut!(sib)) }, unsafe { &mut (*core::ptr::addr_of_mut!(parent)) }) {
break;
}
}
mas_wmb_replace(mas, unsafe { &mut (*core::ptr::addr_of_mut!(cp)) });
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__3568: *const u8 = (b"mas_wr_rebalance\0".as_ptr() as *mut u8) as *const u8;

// mas_wr_store_entry() - Internal call to store a value
// @wr_mas: The maple write state
fn mas_wr_store_entry(wr_mas: *mut ma_wr_state) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
match (unsafe { (*mas).store_type }) as u32 {
3 => {
let _r_a_p__v: u64 = (unsafe { (*wr_mas).entry }) as u64;
loop {
kcsan_release!();
loop {
while false {
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
loop {
while false {
}
__WRITE_ONCE!(*unsafe { (*wr_mas).slots }.wrapping_offset(((unsafe { (*mas).offset }) as i32) as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
if (!unsafe { (*wr_mas).entry }.is_null() as i32) ^ (!unsafe { (*wr_mas).content }.is_null() as i32) != 0 {
mas_update_gap(mas);
}
}
7 => {
mas_wr_append(wr_mas);
}
9 => {
mas_wr_slot_store(wr_mas);
}
8 => {
mas_wr_node_store(wr_mas);
}
4 => {
mas_wr_spanning_store(wr_mas);
}
5 => {
mas_wr_split(wr_mas);
}
6 => {
mas_wr_rebalance(wr_mas);
}
1 => {
mas_new_root(mas, unsafe { (*wr_mas).entry });
}
2 => {
mas_store_root(mas, unsafe { (*wr_mas).entry });
}
0 => {
loop {
if unlikely__2!(1i32) != 0 {
loop {
unsafe {
core::arch::asm!("787: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 787b - ., 3; .popsection\n# {0}", const 787 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mas_wr_store_entry_8, sym __slate_asm_string_mas_wr_store_entry_9, const 3624 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
}
_ => {
}
}
}

fn mas_wr_prealloc_setup(wr_mas: *mut ma_wr_state) {
let mut mas: *mut ma_state = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
'__join_1: {
mas = (*wr_mas).mas;
if !mas_is_active(mas) {
if mas_is_start(mas as *const ma_state) {
break '__join_0;
} else {
if unlikely!(mas_is_paused(mas as *const ma_state)) != 0 {
break '__join_1;
} else {
if unlikely!(mas_is_none(mas as *const ma_state)) != 0 {
break '__join_1;
} else {
if unlikely!(mas_is_overflow(mas)) != 0 {
break '__join_1;
} else {
if unlikely!(mas_is_underflow(mas)) != 0 {
break '__join_1;
}
}
}
}
}
}
// A less strict version of mas_is_span_wr() where we allow spanning
// writes within this node.  This is to stop partial walks in
// mas_prealloc() from being reset.
if !((*mas).last > (*mas).max) {
if !unsafe { (*wr_mas).entry }.is_null() {
break '__join_0;
} else {
if !(mte_is_leaf((*mas).node as *const maple_enode) && (*mas).last == (*mas).max) {
break '__join_0;
}
}
}
}
mas_reset(mas);
}
(*wr_mas).content = mas_start(mas) as *mut ();
}
}

/// mas_prealloc_calc() - Calculate number of nodes needed for a
/// given store operation
/// @wr_mas: The maple write state
/// @entry: The entry to store into the tree
///
/// Return: Number of nodes required for preallocation.
fn mas_prealloc_calc(wr_mas: *mut ma_wr_state, entry: *mut ()) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let height: u8 = mas_mt_height(mas) as u8;
let mut ret: i32 = (height as i32).wrapping_mul(3).wrapping_add(1);
let delta: u8 = (height as i32).wrapping_sub((unsafe { (*wr_mas).vacant_height }) as i32) as u8;
match (unsafe { (*mas).store_type }) as u32 {
3 | 7 | 9 => {
ret = 0;
}
4 => {
if ((unsafe { (*wr_mas).sufficient_height }) as i32) < ((unsafe { (*wr_mas).vacant_height }) as i32) {
ret = (height as i32).wrapping_sub((unsafe { (*wr_mas).sufficient_height }) as i32).wrapping_mul(3).wrapping_add(1);
} else {
ret = (delta as i32).wrapping_mul(3).wrapping_add(1);
}
}
5 => {
ret = (delta as i32).wrapping_mul(2).wrapping_add(1);
}
6 => {
if ((unsafe { (*wr_mas).sufficient_height }) as i32) < ((unsafe { (*wr_mas).vacant_height }) as i32) {
ret = (height as i32).wrapping_sub((unsafe { (*wr_mas).sufficient_height }) as i32).wrapping_mul(2).wrapping_add(1);
} else {
ret = (delta as i32).wrapping_mul(2).wrapping_add(1);
}
}
8 => {
ret = if mt_in_rcu(unsafe { (*mas).tree }) { 1 } else { 0 };
}
1 => {
ret = 1;
}
2 => {
if likely!((unsafe { (*mas).last }) != 0 || (unsafe { (*mas).index }) != 0) != 0 {
ret = 1;
} else {
if (entry as u64) & 3 == 2 {
ret = 1;
} else {
ret = 0;
}
}
}
0 => {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("788: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 788b - ., 3; .popsection\n# {0}", const 788 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_prealloc_calc_2, sym __slate_asm_string_mas_prealloc_calc_3, const 3722 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("789: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 789b - ., 4; .popsection\n# {0}", const 789 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
}
_ => {
}
}
unsafe {
(*mas).node_request = ret as u64;
}
}

/// mas_wr_store_type() - Determine the store type for a given
/// store operation.
/// @wr_mas: The maple write state
///
/// Return: the type of store needed for the operation
fn mas_wr_store_type(wr_mas: *mut ma_wr_state) -> u32 {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
let mut new_end: u8 = 0;
let mut appending: bool = false;
let mut one_slot: bool = false;
let mut in_rcu: bool = false;
let __v0: bool = mas_is_none(mas as *const ma_state) || mas_is_ptr(mas as *const ma_state);
if __v0 {
return 2;
}
if unlikely!(!mas_wr_walk(wr_mas)) != 0 {
return 4;
}
// At this point, we are at the leaf node that needs to be altered.
mas_wr_end_piv(wr_mas);
if unsafe { (*wr_mas).entry }.is_null() {
mas_wr_extend_null(wr_mas);
}
if (unsafe { (*wr_mas).r_min }) == unsafe { (*mas).index } && (unsafe { (*wr_mas).r_max }) == unsafe { (*mas).last } {
return 3;
}
if unlikely!((unsafe { (*mas).index }) == 0 && (unsafe { (*mas).last }) == !0) != 0 {
return 1;
}
new_end = mas_wr_new_end(wr_mas);
in_rcu = mt_in_rcu(unsafe { (*mas).tree });
appending = ((unsafe { (*mas).offset }) as i32) == ((unsafe { (*mas).end }) as i32);
one_slot = ((unsafe { (*wr_mas).offset_end }) as i32).wrapping_sub((unsafe { (*mas).offset }) as i32) == 1;
// Potential spanning rebalance collapsing a node
if (new_end as i32) < ((unsafe { mt_min_slots[((*wr_mas).r#type as u32) as usize] }) as i32) {
if !mte_is_root((unsafe { (*mas).node }) as *const maple_enode) {
return 6;
}
if !in_rcu {
if appending {
return 7;
} else {
if ((unsafe { (*mas).end }) as i32) == (new_end as i32) && one_slot {
return 9;
}
}
}
return 8;
}
if (new_end as i32) >= ((unsafe { mt_slots[((*wr_mas).r#type as u32) as usize] }) as i32) {
return 5;
}
if !in_rcu && appending {
return 7;
}
if (new_end as i32) == ((unsafe { (*mas).end }) as i32) && (!in_rcu || one_slot) {
return 9;
}
8
}

/// mas_wr_preallocate() - Preallocate enough nodes for a store operation
/// @wr_mas: The maple write state
/// @entry: The entry that will be stored
fn mas_wr_preallocate(wr_mas: *mut ma_wr_state, entry: *mut ()) {
let mas: *mut ma_state = unsafe { (*wr_mas).mas };
mas_wr_prealloc_setup(wr_mas);
unsafe {
(*mas).store_type = mas_wr_store_type(wr_mas);
}
mas_prealloc_calc(wr_mas, entry);
if (unsafe { (*mas).node_request }) == 0 {
return;
}
mas_alloc_nodes(mas, (BIT!(11i32) as u32) | (BIT!(13i32) as u32));
}

/// mas_insert() - Internal call to insert a value
/// @mas: The maple state
/// @entry: The entry to store
///
/// Return: %NULL or the contents that already exists at the requested index
/// otherwise.  The maple state needs to be checked for error conditions.
fn mas_insert(mas: *mut ma_state, entry: *mut ()) -> *mut () {
let mut wr_mas: core::mem::MaybeUninit<ma_wr_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut ma_wr_state = core::ptr::addr_of_mut!(wr_mas) as *mut ma_wr_state;
unsafe {
'__join_0: {
core::ptr::write(__slate_slot_0, {
let mut __t0: ma_wr_state = core::mem::zeroed();
__t0.mas = mas;
__t0.entry = entry;
__t0.content = core::ptr::null_mut::<()>();
__t0.vacant_height = 0u8;
__t0.sufficient_height = 0u8;
__t0
});
// Inserting a new range inserts either 0, 1, or 2 pivots within the
// tree.  If the insert fits exactly into an existing gap with a value
// of NULL, then the slot only needs to be written with the new value.
// If the range being inserted is adjacent to another range, then only a
// single pivot needs to be inserted (as well as writing the entry).  If
// the new range is within a gap but does not touch any other ranges,
// then two pivots need to be inserted: the start - 1, and the end.  As
// usual, the entry must be written.  Most operations require a new node
// to be allocated and replace an existing node to ensure RCU safety,
// when in RCU mode.  The exception to requiring a newly allocated node
// is when inserting at the end of a node (appending).  When done
// carefully, appending can reuse the node in place.
(*__slate_slot_0).content = mas_start(mas) as *mut ();
if (*__slate_slot_0).content.is_null() {
mas_wr_preallocate(core::ptr::addr_of_mut!(*__slate_slot_0), entry);
if mas_is_err(mas) {
return core::ptr::null_mut::<()>();
} else {
// spanning writes always overwrite something
if ((*mas).store_type as u32) != 4 {
// At this point, we are at the leaf node that needs to be altered.
if ((*mas).store_type as u32) != 1 && ((*mas).store_type as u32) != 2 {
(*__slate_slot_0).offset_end = (*mas).offset;
(*__slate_slot_0).end_piv = (*__slate_slot_0).r_max;
if !(*__slate_slot_0).content.is_null() || (*mas).last > (*__slate_slot_0).r_max {
break '__join_0;
}
}
mas_wr_store_entry(core::ptr::addr_of_mut!(*__slate_slot_0));
return (*__slate_slot_0).content;
}
}
}
}
mas_set_err(mas, 17i32.wrapping_neg() as i64);
return (*__slate_slot_0).content;
}
}

// mas_alloc_cyclic() - Internal call to find somewhere to store an entry
// @mas: The maple state.
// @startp: Pointer to ID.
// @range_lo: Lower bound of range to search.
// @range_hi: Upper bound of range to search.
// @entry: The entry to store.
// @next: Pointer to next ID to allocate.
// @gfp: The GFP_FLAGS to use for allocations.
//
// Return: 0 if the allocation succeeded without wrapping, 1 if the
// allocation succeeded after wrapping, or -EBUSY if there are no
// free entries.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_alloc_cyclic_793: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, *mut u64, *mut (), u64, u64, *mut u64, u32) -> i32>, *mut ()>(Some(mas_alloc_cyclic)) };

#[inline(always)]
fn mas_rewalk(mas: *mut ma_state, index: u64) {
unsafe {
loop {
mas_set(mas, index);
mas_state_walk(mas);
if !mas_is_start(mas as *const ma_state) {
break;
}
}
}
}

#[inline(always)]
fn mas_rewalk_if_dead(mas: *mut ma_state, node: *mut maple_node, index: u64) -> bool {
if unlikely!(ma_dead_node(node as *const maple_node)) != 0 {
mas_rewalk(mas, index);
return true;
}
false
}

/// mas_prev_node() - Find the prev non-null entry at the same level in the
/// tree.  The prev value will be mas->node[mas->offset] or the status will be
/// ma_none.
/// @mas: The maple state
/// @min: The lower limit to search
///
/// The prev node value will be mas->node[mas->offset] or the status will be
/// ma_none.
/// Return: 1 if the node is dead, 0 otherwise.
fn mas_prev_node(mas: *mut ma_state, min: u64) -> i32 {
let mut max: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(max) as *mut u64;
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut level: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(level) as *mut i32;
let mut offset: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(offset) as *mut i32;
let mut mt: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(mt) as *mut u32;
unsafe {
'__join_2: {
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_8 = mas_mn(mas as *const ma_state);
if (*mas).min != 0 {
*__slate_slot_6 = unsafe { (*mas).min }.wrapping_sub(1);
if *__slate_slot_6 >= min {
*__slate_slot_10 = 0;
loop {
if ma_is_root(*__slate_slot_8) {
break '__join_2;
} else {
// Walk up.
if unlikely__2!(mas_ascend(mas)) != 0 {
return 1;
} else {
*__slate_slot_11 = (*mas).offset as i32;
let __v5: i32 = *__slate_slot_10;
*__slate_slot_10 = __v5.wrapping_add(1);
*__slate_slot_8 = mas_mn(mas as *const ma_state);
if *__slate_slot_11 != 0 {
break;
}
}
}
}
let __v3: i32 = *__slate_slot_11;
*__slate_slot_11 = __v3.wrapping_sub(1);
*__slate_slot_12 = mte_node_type((*mas).node as *const maple_enode);
while *__slate_slot_10 > 1 {
let __v1: i32 = *__slate_slot_10;
*__slate_slot_10 = __v1.wrapping_sub(1);
*__slate_slot_9 = ma_slots(*__slate_slot_8, *__slate_slot_12);
(*mas).node = mas_slot(mas, *__slate_slot_9, *__slate_slot_11 as u8) as *mut maple_enode;
if unlikely!(ma_dead_node(*__slate_slot_8 as *const maple_node)) != 0 {
return 1;
} else {
*__slate_slot_12 = mte_node_type((*mas).node as *const maple_enode);
*__slate_slot_8 = mas_mn(mas as *const ma_state);
*__slate_slot_7 = ma_pivots(*__slate_slot_8, *__slate_slot_12);
*__slate_slot_11 = ma_data_end(*__slate_slot_8, *__slate_slot_12, *__slate_slot_7, *__slate_slot_6) as i32;
if unlikely!(ma_dead_node(*__slate_slot_8 as *const maple_node)) != 0 {
return 1;
}
}
}
*__slate_slot_9 = ma_slots(*__slate_slot_8, *__slate_slot_12);
(*mas).node = mas_slot(mas, *__slate_slot_9, *__slate_slot_11 as u8) as *mut maple_enode;
*__slate_slot_7 = ma_pivots(*__slate_slot_8, *__slate_slot_12);
if unlikely!(ma_dead_node(*__slate_slot_8 as *const maple_node)) != 0 {
return 1;
} else {
if likely__3!(*__slate_slot_11) != 0 {
(*mas).min = unsafe { *(*__slate_slot_7).wrapping_offset((*__slate_slot_11).wrapping_sub(1) as isize) }.wrapping_add(1);
}
(*mas).max = *__slate_slot_6;
(*mas).offset = mas_data_end(mas);
if unlikely!(mte_dead_node((*mas).node as *const maple_enode)) != 0 {
return 1;
} else {
(*mas).end = (*mas).offset;
return 0;
}
}
}
}
}
if unlikely!(ma_dead_node(*__slate_slot_8 as *const maple_node)) != 0 {
return 1;
} else {
(*mas).status = 6;
return 0;
}
}
}

/// mas_prev_slot() - Get the entry in the previous slot
///
/// @mas: The maple state
/// @min: The minimum starting range
/// @empty: Can be empty
///
/// Return: The entry in the previous slot which is possibly NULL
fn mas_prev_slot(mas: *mut ma_state, min: u64, empty: bool) -> *mut () {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut save_point: u64 = unsafe { core::mem::zeroed() };
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
let mut r#type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(r#type) as *mut u32;
let mut pivot: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(pivot) as *mut u64;
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
unsafe {
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
save_point = (*mas).index;
'__join_0: {
'__loop_22: loop {
*__slate_slot_7 = mas_mn(mas as *const ma_state);
*__slate_slot_9 = mte_node_type((*mas).node as *const maple_enode);
*__slate_slot_8 = ma_pivots(*__slate_slot_7, *__slate_slot_9);
if unlikely!(mas_rewalk_if_dead(mas, *__slate_slot_7, save_point)) == 0 {
if (*mas).min <= min {
*__slate_slot_10 = mas_safe_min(mas, *__slate_slot_8, (*mas).offset);
if unlikely!(mas_rewalk_if_dead(mas, *__slate_slot_7, save_point)) != 0 {
continue '__loop_22;
} else {
if *__slate_slot_10 <= min {
break '__join_0;
}
}
}
loop {
if likely__5!(unsafe { (*mas).offset }) != 0 {
(*mas).offset = ((*mas).offset as i32).wrapping_sub(1) as u8;
(*mas).last = unsafe { (*mas).index }.wrapping_sub(1);
(*mas).index = mas_safe_min(mas, *__slate_slot_8, (*mas).offset);
} else {
if (*mas).index <= min {
break '__join_0;
} else {
if mas_prev_node(mas, min) != 0 {
break;
} else {
__ret_warn_on = !(!mas_is_underflow(mas)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("794: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 794b - ., 3; .popsection\n# {0}", const 794 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_prev_slot_13, sym __slate_asm_string_mas_prev_slot_14, const 4073 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("795: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 795b - ., 4; .popsection\n# {0}", const 795 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v5: i64 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
break '__loop_22;
} else {
(*mas).last = (*mas).max;
*__slate_slot_7 = mas_mn(mas as *const ma_state);
*__slate_slot_9 = mte_node_type((*mas).node as *const maple_enode);
*__slate_slot_8 = ma_pivots(*__slate_slot_7, *__slate_slot_9);
(*mas).index = unsafe { *(*__slate_slot_8).wrapping_offset(((*mas).offset as i32).wrapping_sub(1) as isize) }.wrapping_add(1);
}
}
}
}
*__slate_slot_11 = ma_slots(*__slate_slot_7, *__slate_slot_9);
*__slate_slot_12 = mas_slot(mas, *__slate_slot_11, (*mas).offset);
if unlikely!(mas_rewalk_if_dead(mas, *__slate_slot_7, save_point)) != 0 {
continue '__loop_22;
} else {
if likely__4!(*__slate_slot_12) != 0 {
return *__slate_slot_12;
} else {
if !empty {
if (*mas).index <= min {
break '__join_0;
}
} else {
return *__slate_slot_12;
}
}
}
}
mas_rewalk(mas, save_point);
}
}
return core::ptr::null_mut::<()>();
}
(*mas).status = 6;
return core::ptr::null_mut::<()>();
}
}

/// mas_next_node() - Get the next node at the same level in the tree.
/// @mas: The maple state
/// @node: The maple node
/// @max: The maximum pivot value to check.
///
/// The next value will be mas->node[mas->offset] or the status will have
/// overflowed.
/// Return: 1 on dead node, 0 otherwise.
fn mas_next_node(mas: *mut ma_state, mut node: *mut maple_node, max: u64) -> i32 {
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut mt: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(mt) as *mut u32;
let mut node_end: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u8 = core::ptr::addr_of_mut!(node_end) as *mut u8;
let mut level: i32 = unsafe { core::mem::zeroed() };
let mut tmp: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut maple_node = core::ptr::addr_of_mut!(tmp) as *mut *mut maple_node;
let mut enode: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut maple_enode = core::ptr::addr_of_mut!(enode) as *mut *mut maple_enode;
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
let mut min: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(min) as *mut u64;
unsafe {
'__join_2: {
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
level = 0;
let __slate_auto_bytes_4 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_5 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if (*mas).max < max {
*__slate_slot_14 = unsafe { (*mas).max }.wrapping_add(1);
level = 0;
loop {
if ma_is_root(node) {
break '__join_2;
} else {
// Walk up.
if unlikely__2!(mas_ascend(mas)) != 0 {
return 1;
} else {
level = level.wrapping_add(1);
node = mas_mn(mas as *const ma_state);
*__slate_slot_8 = mte_node_type((*mas).node as *const maple_enode);
*__slate_slot_13 = ma_pivots(node, *__slate_slot_8);
*__slate_slot_9 = ma_data_end(node, *__slate_slot_8, *__slate_slot_13, (*mas).max);
if unlikely!(ma_dead_node(node as *const maple_node)) != 0 {
return 1;
} else {
if unlikely!(((*mas).offset as i32) == (*__slate_slot_9 as i32)) == 0 {
break;
}
}
}
}
}
*__slate_slot_7 = ma_slots(node, *__slate_slot_8);
(*mas).offset = ((*mas).offset as i32).wrapping_add(1) as u8;
*__slate_slot_12 = mas_slot(mas, *__slate_slot_7, (*mas).offset) as *mut maple_enode;
if unlikely!(ma_dead_node(node as *const maple_node)) != 0 {
return 1;
} else {
if level > 1 {
(*mas).offset = 0;
}
while unlikely!(level > 1) != 0 {
level = level.wrapping_sub(1);
(*mas).node = *__slate_slot_12;
node = mas_mn(mas as *const ma_state);
*__slate_slot_8 = mte_node_type((*mas).node as *const maple_enode);
*__slate_slot_7 = ma_slots(node, *__slate_slot_8);
*__slate_slot_12 = mas_slot(mas, *__slate_slot_7, 0u8) as *mut maple_enode;
if unlikely!(ma_dead_node(node as *const maple_node)) != 0 {
return 1;
}
}
if (*mas).offset == 0 {
*__slate_slot_13 = ma_pivots(node, *__slate_slot_8);
}
(*mas).max = mas_safe_pivot(mas as *const ma_state, *__slate_slot_13, (*mas).offset, *__slate_slot_8);
*__slate_slot_11 = mte_to_node(*__slate_slot_12 as *const maple_enode);
*__slate_slot_8 = mte_node_type(*__slate_slot_12 as *const maple_enode);
*__slate_slot_13 = ma_pivots(*__slate_slot_11, *__slate_slot_8);
(*mas).end = ma_data_end(*__slate_slot_11, *__slate_slot_8, *__slate_slot_13, (*mas).max);
if unlikely!(ma_dead_node(node as *const maple_node)) != 0 {
return 1;
} else {
(*mas).node = *__slate_slot_12;
(*mas).min = *__slate_slot_14;
return 0;
}
}
}
}
if unlikely!(ma_dead_node(node as *const maple_node)) != 0 {
return 1;
} else {
(*mas).status = 5;
return 0;
}
}
}

/// mas_next_slot() - Get the entry in the next slot
///
/// @mas: The maple state
/// @max: The maximum starting range
/// @empty: Can be empty
///
/// Return: The entry in the next slot which is possibly NULL
fn mas_next_slot(mas: *mut ma_state, max: u64, empty: bool) -> *mut () {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
let mut save_point: u64 = unsafe { core::mem::zeroed() };
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
let mut r#type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(r#type) as *mut u32;
let mut pivot: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(pivot) as *mut u64;
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut __slate_state: usize = 32usize;
unsafe {
loop {
match __slate_state {
0 => {
return *__slate_slot_6;
}
1 => {
(*mas).index = unsafe { (*mas).last }.wrapping_add(1);
__slate_state = 10usize;
continue;
}
2 => {
(*mas).status = 5;
return core::ptr::null_mut::<()>();
}
3 => {
if (*mas).last >= max {
__slate_state = 2usize;
continue;
} else {
__slate_state = 1usize;
continue;
}
}
4 => {
if !empty {
__slate_state = 3usize;
continue;
} else {
__slate_state = 0usize;
continue;
}
}
5 => {
return *__slate_slot_6;
}
6 => {
if !(*__slate_slot_6).is_null() {
__slate_state = 5usize;
continue;
} else {
__slate_state = 4usize;
continue;
}
}
7 => {
*__slate_slot_12 = ma_slots(*__slate_slot_8, *__slate_slot_9);
*__slate_slot_6 = mt_slot((*mas).tree as *const maple_tree, *__slate_slot_12, (*mas).offset);
if unlikely!(mas_rewalk_if_dead(mas, *__slate_slot_8, save_point)) != 0 {
__slate_state = 31usize;
continue;
} else {
__slate_state = 6usize;
continue;
}
}
8 => {
(*mas).last = *(*__slate_slot_11).wrapping_offset(((*mas).offset as i32) as isize);
__slate_state = 7usize;
continue;
}
9 => {
(*mas).last = (*mas).max;
__slate_state = 7usize;
continue;
}
10 => {
(*mas).offset = ((*mas).offset as i32).wrapping_add(1) as u8;
if likely!(((*mas).offset as i32) < ((*mas).end as i32)) != 0 {
__slate_state = 8usize;
continue;
} else {
__slate_state = 9usize;
continue;
}
}
11 => {
(*mas).index = unsafe { *(*__slate_slot_11).wrapping_offset(((*mas).offset as i32) as isize) }.wrapping_add(1);
__slate_state = 10usize;
continue;
}
12 => {
(*mas).offset = 0;
(*mas).index = (*mas).min;
*__slate_slot_8 = mas_mn(mas as *const ma_state);
*__slate_slot_9 = mte_node_type((*mas).node as *const maple_enode);
*__slate_slot_11 = ma_pivots(*__slate_slot_8, *__slate_slot_9);
(*mas).last = *(*__slate_slot_11).wrapping_offset(0 as isize);
__slate_state = 7usize;
continue;
}
13 => {
return core::ptr::null_mut::<()>();
}
14 => {
if __ret_warn_on != 0 {
__slate_state = 13usize;
continue;
} else {
__slate_state = 12usize;
continue;
}
}
15 => {
core::arch::asm!("797: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 797b - ., 4; .popsection\n# {0}", const 797 as i32, options(nostack, att_syntax));
if false {
__slate_state = 17usize;
continue;
} else {
__slate_state = 14usize;
continue;
}
}
16 => {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_next_slot_13, sym __slate_asm_string_mas_next_slot_14, const 4253 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if false {
__slate_state = 16usize;
continue;
} else {
__slate_state = 15usize;
continue;
}
}
17 => {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("796: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 796b - ., 3; .popsection\n# {0}", const 796 as i32, options(nostack, att_syntax));
__slate_state = 16usize;
continue;
}
18 => {
__ret_warn_on = !(!mas_is_overflow(mas)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
__slate_state = 17usize;
continue;
} else {
__slate_state = 14usize;
continue;
}
}
19 => {
mas_rewalk(mas, save_point);
__slate_state = 31usize;
continue;
}
20 => {
if mas_next_node(mas, *__slate_slot_8, max) != 0 {
__slate_state = 19usize;
continue;
} else {
__slate_state = 18usize;
continue;
}
}
21 => {
(*mas).status = 5;
return core::ptr::null_mut::<()>();
}
22 => {
if (*mas).last >= max {
__slate_state = 21usize;
continue;
} else {
__slate_state = 20usize;
continue;
}
}
23 => {
if likely!(((*mas).offset as i32) < ((*mas).end as i32)) != 0 {
__slate_state = 11usize;
continue;
} else {
__slate_state = 22usize;
continue;
}
}
24 => {
// Was at the limit, next will extend beyond
(*mas).status = 5;
return core::ptr::null_mut::<()>();
}
25 => {
if *__slate_slot_10 >= max {
__slate_state = 24usize;
continue;
} else {
__slate_state = 23usize;
continue;
}
}
26 => {
if unlikely!(mas_rewalk_if_dead(mas, *__slate_slot_8, save_point)) != 0 {
__slate_state = 31usize;
continue;
} else {
__slate_state = 25usize;
continue;
}
}
27 => {
*__slate_slot_10 = *(*__slate_slot_11).wrapping_offset(((*mas).offset as i32) as isize);
__slate_state = 26usize;
continue;
}
28 => {
*__slate_slot_10 = (*mas).max;
__slate_state = 26usize;
continue;
}
29 => {
if likely!(((*mas).offset as i32) < ((*mas).end as i32)) != 0 {
__slate_state = 27usize;
continue;
} else {
__slate_state = 28usize;
continue;
}
}
30 => {
if (*mas).max >= max {
__slate_state = 29usize;
continue;
} else {
__slate_state = 23usize;
continue;
}
}
31 => {
*__slate_slot_8 = mas_mn(mas as *const ma_state);
*__slate_slot_9 = mte_node_type((*mas).node as *const maple_enode);
*__slate_slot_11 = ma_pivots(*__slate_slot_8, *__slate_slot_9);
if unlikely!(mas_rewalk_if_dead(mas, *__slate_slot_8, save_point)) != 0 {
__slate_state = 31usize;
continue;
} else {
__slate_state = 30usize;
continue;
}
}
32 => {
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
save_point = (*mas).last;
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
__slate_state = 31usize;
continue;
}
_ => {
core::hint::unreachable_unchecked();
}
}
}
}
unsafe { core::mem::zeroed() }
}

/// mas_rev_awalk() - Internal function.  Reverse allocation walk.  Find the
/// highest gap address of a given size in a given node and descend.
/// @mas: The maple state
/// @size: The needed size.
///
/// Return: True if found in a leaf, false otherwise.
fn mas_rev_awalk(mas: *mut ma_state, size: u64, gap_min: &mut u64, gap_max: &mut u64) -> bool {
let mut offset: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u8 = core::ptr::addr_of_mut!(offset) as *mut u8;
let mut min: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(min) as *mut u64;
let mut max: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u64 = core::ptr::addr_of_mut!(max) as *mut u64;
let mut gap: u64 = unsafe { core::mem::zeroed() };
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut gaps: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut u64 = core::ptr::addr_of_mut!(gaps) as *mut *mut u64;
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
let mut node: *mut maple_node = unsafe { core::mem::zeroed() };
let mut r#type: u32 = unsafe { core::mem::zeroed() };
unsafe {
r#type = mte_node_type((*mas).node as *const maple_enode);
node = mas_mn(mas as *const ma_state);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
gap = 0;
let __slate_auto_bytes_3 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if unlikely!(mas_is_err(mas)) != 0 {
return true;
} else {
if ma_is_dense(r#type) {
// dense nodes.
(*mas).offset = unsafe { (*mas).index }.wrapping_sub((*mas).min) as u8;
return true;
} else {
*__slate_slot_12 = ma_pivots(node, r#type);
*__slate_slot_10 = ma_slots(node, r#type);
*__slate_slot_11 = ma_gaps(node, r#type);
*__slate_slot_6 = (*mas).offset;
*__slate_slot_7 = mas_safe_min(mas, *__slate_slot_12, *__slate_slot_6);
// Skip out of bounds.
while (*mas).last < *__slate_slot_7 {
let __v4: u8 = (*__slate_slot_6 as i32).wrapping_sub(1) as u8;
*__slate_slot_6 = __v4;
*__slate_slot_7 = mas_safe_min(mas, *__slate_slot_12, __v4);
}
*__slate_slot_8 = mas_safe_pivot(mas as *const ma_state, *__slate_slot_12, *__slate_slot_6, r#type);
'__join_0: {
'__join_6: {
loop {
if (*mas).index <= *__slate_slot_8 {
gap = 0;
if !(*__slate_slot_11).is_null() {
gap = *(*__slate_slot_11).wrapping_offset((*__slate_slot_6 as i32) as isize);
} else {
if mas_slot(mas, *__slate_slot_10, *__slate_slot_6).is_null() {
gap = (*__slate_slot_8).wrapping_sub(*__slate_slot_7).wrapping_add(1);
}
}
if gap != 0 {
if size <= gap && size <= unsafe { (*mas).last }.wrapping_sub(*__slate_slot_7).wrapping_add(1) {
break '__join_6;
} else {
if (*__slate_slot_11).is_null() {
// Skip the next slot, it cannot be a gap.
if (*__slate_slot_6 as i32) < 2 {
break;
} else {
*__slate_slot_6 = (*__slate_slot_6 as i32).wrapping_sub(2) as u8;
*__slate_slot_8 = *(*__slate_slot_12).wrapping_offset((*__slate_slot_6 as i32) as isize);
*__slate_slot_7 = mas_safe_min(mas, *__slate_slot_12, *__slate_slot_6);
continue;
}
}
}
}
if *__slate_slot_6 == 0 {
break;
} else {
*__slate_slot_6 = (*__slate_slot_6 as i32).wrapping_sub(1) as u8;
*__slate_slot_8 = (*__slate_slot_7).wrapping_sub(1);
*__slate_slot_7 = mas_safe_min(mas, *__slate_slot_12, *__slate_slot_6);
}
} else {
break '__join_6;
}
}
if !mte_is_root((*mas).node as *const maple_enode) {
return false;
} else {
break '__join_0;
}
}
if unlikely!((*mas).index > *__slate_slot_8 || size.wrapping_sub(1) > (*__slate_slot_8).wrapping_sub((*mas).index)) == 0 {
if unlikely!(ma_is_leaf(r#type)) != 0 {
(*mas).offset = *__slate_slot_6;
*(gap_min as *mut u64) = *__slate_slot_7;
*(gap_max as *mut u64) = (*__slate_slot_7).wrapping_add(gap).wrapping_sub(1);
return true;
} else {
// descend, only happens under lock.
(*mas).node = mas_slot(mas, *__slate_slot_10, *__slate_slot_6) as *mut maple_enode;
(*mas).min = *__slate_slot_7;
(*mas).max = *__slate_slot_8;
(*mas).offset = mas_data_end(mas);
return false;
}
}
}
mas_set_err(mas, 16i32.wrapping_neg() as i64);
return false;
}
}
}
}

fn mas_anode_descend(mas: *mut ma_state, size: u64) -> bool {
let mut __UNIQUE_ID_y__802: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__801: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__799: u64 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__798: u64 = unsafe { core::mem::zeroed() };
let mut __v7: u64 = unsafe { core::mem::zeroed() };
let mut found: bool = unsafe { core::mem::zeroed() };
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
let mut pivots: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut u64 = core::ptr::addr_of_mut!(pivots) as *mut *mut u64;
let mut gaps: core::mem::MaybeUninit<*mut u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut u64 = core::ptr::addr_of_mut!(gaps) as *mut *mut u64;
let mut data_end: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u8 = core::ptr::addr_of_mut!(data_end) as *mut u8;
let mut offset: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u8 = core::ptr::addr_of_mut!(offset) as *mut u8;
let mut gap: u64 = unsafe { core::mem::zeroed() };
let mut min: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u64 = core::ptr::addr_of_mut!(min) as *mut u64;
let mut pivot: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u64 = core::ptr::addr_of_mut!(pivot) as *mut u64;
let mut r#type: u32 = unsafe { core::mem::zeroed() };
unsafe {
r#type = mte_node_type((*mas).node as *const maple_enode);
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
gap = 0;
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_4 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
found = false;
if ma_is_dense(r#type) {
(*mas).offset = unsafe { (*mas).index }.wrapping_sub((*mas).min) as u8;
return true;
} else {
*__slate_slot_9 = mas_mn(mas as *const ma_state);
*__slate_slot_11 = ma_pivots(*__slate_slot_9, r#type);
*__slate_slot_10 = ma_slots(*__slate_slot_9, r#type);
*__slate_slot_12 = ma_gaps(*__slate_slot_9, r#type);
*__slate_slot_14 = (*mas).offset;
*__slate_slot_16 = mas_safe_min(mas, *__slate_slot_11, *__slate_slot_14);
*__slate_slot_13 = ma_data_end(*__slate_slot_9, r#type, *__slate_slot_11, (*mas).max);
'__join_0: {
'__loop_1: loop {
if (*__slate_slot_14 as i32) <= (*__slate_slot_13 as i32) {
'__join_4: {
*__slate_slot_17 = mas_safe_pivot(mas as *const ma_state, *__slate_slot_11, *__slate_slot_14, r#type);
// Not within lower bounds
if !((*mas).index > *__slate_slot_17) {
if !(*__slate_slot_12).is_null() {
gap = *(*__slate_slot_12).wrapping_offset((*__slate_slot_14 as i32) as isize);
} else {
if mas_slot(mas, *__slate_slot_10, *__slate_slot_14).is_null() {
__UNIQUE_ID_x__798 = *__slate_slot_17;
__UNIQUE_ID_y__799 = (*mas).last;
while false {
}
__v7 = if __UNIQUE_ID_x__798 < __UNIQUE_ID_y__799 { __UNIQUE_ID_x__798 } else { __UNIQUE_ID_y__799 };
__UNIQUE_ID_x__801 = (*mas).index;
__UNIQUE_ID_y__802 = *__slate_slot_16;
while false {
}
gap = __v7.wrapping_sub(if __UNIQUE_ID_x__801 > __UNIQUE_ID_y__802 { __UNIQUE_ID_x__801 } else { __UNIQUE_ID_y__802 }).wrapping_add(1);
} else {
break '__join_4;
}
}
if gap >= size {
break '__loop_1;
}
}
}
*__slate_slot_16 = (*__slate_slot_17).wrapping_add(1);
if (*mas).last <= *__slate_slot_17 {
mas_set_err(mas, 16i32.wrapping_neg() as i64);
return true;
} else {
*__slate_slot_14 = (*__slate_slot_14 as i32).wrapping_add(1) as u8;
}
} else {
break '__join_0;
}
}
if ma_is_leaf(r#type) {
found = true;
} else {
(*mas).node = mas_slot(mas, *__slate_slot_10, *__slate_slot_14) as *mut maple_enode;
(*mas).min = *__slate_slot_16;
(*mas).max = *__slate_slot_17;
*__slate_slot_14 = 0;
}
}
(*mas).offset = *__slate_slot_14;
return found;
}
}
}

/// mas_walk() - Search for @mas->index in the tree.
/// @mas: The maple state.
///
/// mas->index and mas->last will be set to the range if there is a value.  If
/// mas->status is ma_none, reset to ma_start
///
/// Return: the entry at the location or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_walk")]
extern "C" fn mas_walk(mas: *mut ma_state) -> *mut () {
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
mas_may_init_lock_check(mas);
mas_make_walkable(mas);
loop {
*__slate_slot_0 = mas_state_walk(mas);
if !mas_is_start(mas as *const ma_state) {
break;
}
}
if mas_is_none(mas as *const ma_state) {
(*mas).index = 0;
(*mas).last = !0;
} else {
if mas_is_ptr(mas as *const ma_state) {
if (*mas).index == 0 {
(*mas).last = 0;
return *__slate_slot_0;
} else {
(*mas).index = 1;
(*mas).last = !0;
(*mas).status = 3;
return core::ptr::null_mut::<()>();
}
}
}
return *__slate_slot_0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_walk_804: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state) -> *mut ()>, *mut ()>(Some(mas_walk)) };

fn mas_rewind_node(mas: *mut ma_state) -> bool {
let mut slot: u8 = 0;
loop {
if mte_is_root((unsafe { (*mas).node }) as *const maple_enode) {
slot = unsafe { (*mas).offset };
if slot == 0 {
return false;
}
} else {
mas_ascend(mas);
slot = unsafe { (*mas).offset };
}
if slot != 0 {
break;
}
}
let __v2: u8 = (slot as i32).wrapping_sub(1) as u8;
slot = __v2;
unsafe {
(*mas).offset = __v2;
}
true
}

/// mas_skip_node() - Internal function.  Skip over a node.
/// @mas: The maple state.
///
/// Return: true if there is another node, false otherwise.
fn mas_skip_node(mas: *mut ma_state) -> bool {
if mas_is_err(mas) {
return false;
}
loop {
if mte_is_root((unsafe { (*mas).node }) as *const maple_enode) {
if ((unsafe { (*mas).offset }) as i32) >= (mas_data_end(mas) as i32) {
mas_set_err(mas, 16i32.wrapping_neg() as i64);
return false;
}
} else {
mas_ascend(mas);
}
if ((unsafe { (*mas).offset }) as i32) < (mas_data_end(mas) as i32) {
break;
}
}
unsafe {
(*mas).offset = ((*mas).offset as i32).wrapping_add(1) as u8;
}
true
}

/// mas_awalk() - Allocation walk.  Search from low address to high, for a gap of
/// @size
/// @mas: The maple state
/// @size: The size of the gap required
///
/// Search between @mas->index and @mas->last for a gap of @size.
fn mas_awalk(mas: *mut ma_state, size: u64) {
let mut last: *mut maple_enode = core::ptr::null_mut::<maple_enode>();
// There are 4 options:
// go to child (descend)
// go back to parent (ascend)
// no gap found. (return, error == -EBUSY)
// found the gap. (return)
loop {
let __v0: bool = !mas_is_err(mas) && !mas_anode_descend(mas, size);
if !__v0 {
break;
}
if last == unsafe { (*mas).node } {
mas_skip_node(mas);
} else {
last = unsafe { (*mas).node };
}
}
}

/// mas_sparse_area() - Internal function.  Return upper or lower limit when
/// searching for a gap in an empty tree.
/// @mas: The maple state
/// @min: the minimum range
/// @max: The maximum range
/// @size: The size of the gap
/// @fwd: Searching forward or back
fn mas_sparse_area(mas: *mut ma_state, mut min: u64, max: u64, size: u64, fwd: bool) -> i32 {
if unlikely!(mas_is_none(mas as *const ma_state)) == 0 && min == 0 {
min = min.wrapping_add(1);
// At this time, min is increased, we need to recheck whether
// the size is satisfied.
if min > max || max.wrapping_sub(min).wrapping_add(1) < size {
return 16i32.wrapping_neg();
}
}
// mas_is_ptr
if fwd {
unsafe {
(*mas).index = min;
}
unsafe {
(*mas).last = min.wrapping_add(size).wrapping_sub(1);
}
} else {
unsafe {
(*mas).last = max;
}
unsafe {
(*mas).index = max.wrapping_sub(size).wrapping_add(1);
}
}
0
}

/// mas_empty_area() - Get the lowest address within the range that is
/// sufficient for the size requested.
/// @mas: The maple state
/// @min: The lowest value of the range
/// @max: The highest value of the range
/// @size: The size needed
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_empty_area")]
extern "C" fn mas_empty_area(mas: *mut ma_state, mut min: u64, max: u64, size: u64) -> i32 {
let mut offset: u8 = 0;
let mut mt: u32 = 0;
if min > max {
return 22i32.wrapping_neg();
}
if size == 0 || max.wrapping_sub(min) < size.wrapping_sub(1) {
return 22i32.wrapping_neg();
}
if mas_is_start(mas as *const ma_state) {
mas_start(mas);
} else {
if ((unsafe { (*mas).offset }) as i32) >= 2 {
unsafe {
(*mas).offset = ((*mas).offset as i32).wrapping_sub(2) as u8;
}
} else {
if !mas_skip_node(mas) {
return 16i32.wrapping_neg();
}
}
}
// Empty set
let __v3: bool = mas_is_none(mas as *const ma_state) || mas_is_ptr(mas as *const ma_state);
if __v3 {
return mas_sparse_area(mas, min, max, size, true);
}
// The start of the window can only be within these values
unsafe {
(*mas).index = min;
}
unsafe {
(*mas).last = max;
}
mas_awalk(mas, size);
if unlikely!(mas_is_err(mas)) != 0 {
return xa_err((unsafe { (*mas).node }) as *mut ());
}
offset = unsafe { (*mas).offset };
let node: *mut maple_node = mas_mn(mas as *const ma_state);
mt = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
let pivots: *mut u64 = ma_pivots(node, mt);
min = mas_safe_min(mas, pivots, offset);
if (unsafe { (*mas).index }) < min {
unsafe {
(*mas).index = min;
}
}
unsafe {
(*mas).last = unsafe { (*mas).index }.wrapping_add(size).wrapping_sub(1);
}
unsafe {
(*mas).end = ma_data_end(node, mt, pivots, (*mas).max);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_empty_area_805: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64, u64, u64) -> i32>, *mut ()>(Some(mas_empty_area)) };

/// mas_empty_area_rev() - Get the highest address within the range that is
/// sufficient for the size requested.
/// @mas: The maple state
/// @min: The lowest value of the range
/// @max: The highest value of the range
/// @size: The size needed
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_empty_area_rev")]
extern "C" fn mas_empty_area_rev(mas: *mut ma_state, mut min: u64, mut max: u64, size: u64) -> i32 {
let mut last: *mut maple_enode = unsafe { (*mas).node };
if min > max {
return 22i32.wrapping_neg();
}
if size == 0 || max.wrapping_sub(min) < size.wrapping_sub(1) {
return 22i32.wrapping_neg();
}
if mas_is_start(mas as *const ma_state) {
mas_start(mas);
} else {
let __v0: bool = ((unsafe { (*mas).offset }) as i32) < 2 && !mas_rewind_node(mas);
if __v0 {
return 16i32.wrapping_neg();
}
}
let __v1: bool = mas_is_none(mas as *const ma_state) || mas_is_ptr(mas as *const ma_state);
if __v1 {
return mas_sparse_area(mas, min, max, size, false);
} else {
if ((unsafe { (*mas).offset }) as i32) >= 2 {
unsafe {
(*mas).offset = ((*mas).offset as i32).wrapping_sub(2) as u8;
}
} else {
unsafe {
(*mas).offset = mas_data_end(mas);
}
}
}
// The start of the window can only be within these values.
unsafe {
(*mas).index = min;
}
unsafe {
(*mas).last = max;
}
while !mas_rev_awalk(mas, size, unsafe { &mut (*core::ptr::addr_of_mut!(min)) }, unsafe { &mut (*core::ptr::addr_of_mut!(max)) }) {
if last == unsafe { (*mas).node } {
if !mas_rewind_node(mas) {
return 16i32.wrapping_neg();
}
} else {
last = unsafe { (*mas).node };
}
}
if mas_is_err(mas) {
return xa_err((unsafe { (*mas).node }) as *mut ());
}
if unlikely!(((unsafe { (*mas).offset }) as i32) == 31) != 0 {
return 16i32.wrapping_neg();
}
// Trim the upper limit to the max.
if max < unsafe { (*mas).last } {
unsafe {
(*mas).last = max;
}
}
unsafe {
(*mas).index = unsafe { (*mas).last }.wrapping_sub(size).wrapping_add(1);
}
unsafe {
(*mas).end = mas_data_end(mas);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_empty_area_rev_806: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64, u64, u64) -> i32>, *mut ()>(Some(mas_empty_area_rev)) };

// mte_dead_leaves() - Mark all leaves of a node as dead.
// @enode: the encoded node
// @mt: the maple tree
// @slots: Pointer to the slot array
//
// Must hold the write lock.
//
// Return: The number of leaves marked as dead.
fn mte_dead_leaves(enode: *mut maple_enode, mt: *mut maple_tree, slots: *mut *mut ()) -> u8 {
let mut r#type: u32 = 0;
let mut offset: i32 = 0;
while offset < (mt_slot_count!(enode) as i32) {
let entry: *mut () = mt_slot(mt as *const maple_tree, slots, offset as u8);
r#type = mte_node_type(entry as *const maple_enode);
let node: *mut maple_node = mte_to_node(entry as *const maple_enode);
// Use both node and type to catch LE & BE metadata
if node.is_null() || (r#type as u32) == 0 {
break;
}
mte_set_node_dead(entry as *mut maple_enode);
unsafe {
(*node).__slate_anon_0.__slate_anon_1.r#type = r#type;
}
let _r_a_p__v: u64 = node as u64;
if false && _r_a_p__v == 0 {
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_807() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slots.wrapping_offset(offset as isize), _r_a_p__v as *mut ());
if true {
break;
}
}
} else {
loop {
kcsan_release!();
loop {
loop {
if !__native_word!() {
unsafe { __compiletime_assert_808() };
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
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_809() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE!(*slots.wrapping_offset(offset as isize), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
offset = offset.wrapping_add(1);
}
offset as u8
}

/// mte_dead_walk() - Walk down a dead tree to just before the leaves
/// @enode: The maple encoded node
/// @offset: The starting offset
///
/// Note: This can only be used from the RCU callback context.
fn mte_dead_walk(enode: &mut *mut maple_enode, mut offset: u8) -> *mut *mut () {
let mut next: *mut maple_node = unsafe { core::mem::zeroed() };
let mut slots: *mut *mut () = core::ptr::null_mut::<*mut ()>();
next = mte_to_node((unsafe { *(enode as *mut *mut maple_enode) }) as *const maple_enode);
loop {
unsafe {
*(enode as *mut *mut maple_enode) = ma_enode_ptr!(next);
}
let node: *mut maple_node = mte_to_node((unsafe { *(enode as *mut *mut maple_enode) }) as *const maple_enode);
slots = ma_slots(node, unsafe { (*node).__slate_anon_0.__slate_anon_1.r#type });
next = rcu_dereference_protected!(unsafe { *slots.wrapping_offset((offset as i32) as isize) }, unsafe { lock_is_held(core::ptr::addr_of_mut!(rcu_callback_map) as *const ()) }) as *mut maple_node;
offset = 0;
if !(!ma_is_leaf(unsafe { (*next).__slate_anon_0.__slate_anon_1.r#type })) {
break;
}
}
slots
}

/// mt_free_walk() - Walk & free a tree in the RCU callback context
/// @head: The RCU head that's within the node.
///
/// Note: This can only be used from the RCU callback context.
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mt_free_walk")]
extern "C" fn mt_free_walk(head: *mut callback_head) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut r#type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(r#type) as *mut u32;
let mut offset: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u8 = core::ptr::addr_of_mut!(offset) as *mut u8;
let mut enode: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut maple_enode = core::ptr::addr_of_mut!(enode) as *mut *mut maple_enode;
let mut start: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut maple_node = core::ptr::addr_of_mut!(start) as *mut *mut maple_node;
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
unsafe {
'__join_0: {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_5 = container_of__21!(head, maple_node);
if !ma_is_leaf((*(*__slate_slot_5)).__slate_anon_0.__slate_anon_1.r#type) {
*__slate_slot_4 = *__slate_slot_5;
*__slate_slot_3 = mt_mk_node(*__slate_slot_5 as *const maple_node, (*(*__slate_slot_5)).__slate_anon_0.__slate_anon_1.r#type);
*__slate_slot_6 = mte_dead_walk(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3)), 0u8);
*__slate_slot_5 = mte_to_node(*__slate_slot_3 as *const maple_enode);
loop {
mt_free_bulk((*(*__slate_slot_5)).__slate_anon_0.__slate_anon_1.slot_len as u64, *__slate_slot_6);
*__slate_slot_2 = ((*(*__slate_slot_5)).__slate_anon_0.__slate_anon_1.parent_slot as i32).wrapping_add(1) as u8;
*__slate_slot_3 = (*(*__slate_slot_5)).__slate_anon_0.__slate_anon_1.piv_parent;
if mte_to_node(*__slate_slot_3 as *const maple_enode) == *__slate_slot_5 {
break '__join_0;
} else {
*__slate_slot_1 = mte_node_type(*__slate_slot_3 as *const maple_enode);
*__slate_slot_6 = ma_slots(mte_to_node(*__slate_slot_3 as *const maple_enode), *__slate_slot_1);
if (*__slate_slot_2 as i32) < (mt_slots[(*__slate_slot_1 as u32) as usize] as i32) {
__v0 = !rcu_dereference_protected!(unsafe { *(*__slate_slot_6).wrapping_offset((*__slate_slot_2 as i32) as isize) }, lock_is_held(core::ptr::addr_of_mut!(rcu_callback_map) as *const ())).is_null();
} else {
__v0 = false;
}
if __v0 {
*__slate_slot_6 = mte_dead_walk(&mut (*core::ptr::addr_of_mut!(*__slate_slot_3)), *__slate_slot_2);
}
*__slate_slot_5 = mte_to_node(*__slate_slot_3 as *const maple_enode);
if !(*__slate_slot_5 != *__slate_slot_4 || ((*(*__slate_slot_5)).__slate_anon_0.__slate_anon_1.slot_len as i32) < (*__slate_slot_2 as i32)) {
break;
}
}
}
*__slate_slot_6 = ma_slots(*__slate_slot_5, (*(*__slate_slot_5)).__slate_anon_0.__slate_anon_1.r#type);
mt_free_bulk((*(*__slate_slot_5)).__slate_anon_0.__slate_anon_1.slot_len as u64, *__slate_slot_6);
}
}
kfree(*__slate_slot_5 as *const ());
}
}

fn mte_destroy_descend(enode: &mut *mut maple_enode, mt: *mut maple_tree, mut prev: *mut maple_enode, mut offset: u8) -> *mut *mut () {
let mut next: *mut maple_enode = unsafe { *(enode as *mut *mut maple_enode) };
let mut slots: *mut *mut () = core::ptr::null_mut::<*mut ()>();
let mut r#type: u32 = 0;
let mut next_offset: u8 = 0;
loop {
unsafe {
*(enode as *mut *mut maple_enode) = next;
}
let node: *mut maple_node = mte_to_node((unsafe { *(enode as *mut *mut maple_enode) }) as *const maple_enode);
r#type = mte_node_type((unsafe { *(enode as *mut *mut maple_enode) }) as *const maple_enode);
slots = ma_slots(node, r#type);
next = mt_slot_locked(mt, slots, next_offset) as *mut maple_enode;
if mte_dead_node(next as *const maple_enode) {
let __v1: u8 = (next_offset as i32).wrapping_add(1) as u8;
next_offset = __v1;
next = mt_slot_locked(mt, slots, __v1) as *mut maple_enode;
}
mte_set_node_dead(unsafe { *(enode as *mut *mut maple_enode) });
unsafe {
(*node).__slate_anon_0.__slate_anon_1.r#type = r#type;
}
unsafe {
(*node).__slate_anon_0.__slate_anon_1.piv_parent = prev;
}
unsafe {
(*node).__slate_anon_0.__slate_anon_1.parent_slot = offset;
}
offset = next_offset;
next_offset = 0;
prev = unsafe { *(enode as *mut *mut maple_enode) };
if !(!mte_is_leaf(next as *const maple_enode)) {
break;
}
}
slots
}

fn mt_destroy_walk(mut enode: *mut maple_enode, mt: *mut maple_tree, free: bool) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut tmp: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut maple_enode = core::ptr::addr_of_mut!(tmp) as *mut *mut maple_enode;
let mut parent: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut maple_enode = core::ptr::addr_of_mut!(parent) as *mut *mut maple_enode;
let mut offset: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u8 = core::ptr::addr_of_mut!(offset) as *mut u8;
let mut r#type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(r#type) as *mut u32;
let mut start: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut maple_enode = core::ptr::addr_of_mut!(start) as *mut *mut maple_enode;
let mut node: *mut maple_node = unsafe { core::mem::zeroed() };
let mut slots: core::mem::MaybeUninit<*mut *mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut *mut () = core::ptr::addr_of_mut!(slots) as *mut *mut *mut ();
unsafe {
'__join_3: {
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
node = mte_to_node(enode as *const maple_enode);
let __slate_auto_bytes_4 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
if mte_is_leaf(enode as *const maple_enode) {
mte_set_node_dead(enode);
(*node).__slate_anon_0.__slate_anon_1.r#type = mte_node_type(enode as *const maple_enode);
} else {
*__slate_slot_5 = enode;
*__slate_slot_7 = mte_destroy_descend(&mut (*core::ptr::addr_of_mut!(enode)), mt, *__slate_slot_5, 0u8);
node = mte_to_node(enode as *const maple_enode); // Updated in the above call.
loop {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
(*node).__slate_anon_0.__slate_anon_1.slot_len = mte_dead_leaves(enode, mt, *__slate_slot_7);
if free {
mt_free_bulk((*node).__slate_anon_0.__slate_anon_1.slot_len as u64, *__slate_slot_7);
}
*__slate_slot_3 = ((*node).__slate_anon_0.__slate_anon_1.parent_slot as i32).wrapping_add(1) as u8;
enode = (*node).__slate_anon_0.__slate_anon_1.piv_parent;
if mte_to_node(enode as *const maple_enode) == node {
break '__join_3;
} else {
*__slate_slot_4 = mte_node_type(enode as *const maple_enode);
*__slate_slot_7 = ma_slots(mte_to_node(enode as *const maple_enode), *__slate_slot_4);
if (*__slate_slot_3 as i32) < (mt_slots[(*__slate_slot_4 as u32) as usize] as i32) {
*__slate_slot_1 = mt_slot_locked(mt, *__slate_slot_7, *__slate_slot_3) as *mut maple_enode;
if (mte_node_type(*__slate_slot_1 as *const maple_enode) as u32) != 0 {
__v0 = !mte_to_node(*__slate_slot_1 as *const maple_enode).is_null();
} else {
__v0 = false;
}
if __v0 {
*__slate_slot_2 = enode;
enode = *__slate_slot_1;
*__slate_slot_7 = mte_destroy_descend(&mut (*core::ptr::addr_of_mut!(enode)), mt, *__slate_slot_2, *__slate_slot_3);
}
}
node = mte_to_node(enode as *const maple_enode);
if *__slate_slot_5 == enode {
break;
}
}
}
node = mte_to_node(enode as *const maple_enode);
(*node).__slate_anon_0.__slate_anon_1.slot_len = mte_dead_leaves(enode, mt, *__slate_slot_7);
if free {
mt_free_bulk((*node).__slate_anon_0.__slate_anon_1.slot_len as u64, *__slate_slot_7);
}
}
}
if free {
kfree(node as *const ());
} else {
mt_clear_meta(mt, node, (*node).__slate_anon_0.__slate_anon_1.r#type);
}
}
}

/// mte_destroy_walk() - Free a tree or sub-tree.
/// @enode: the encoded maple node (maple_enode) to start
/// @mt: the tree to free - needed for node types.
///
/// Must hold the write lock.
fn mte_destroy_walk(enode: *mut maple_enode, mt: *mut maple_tree) {
let node: *mut maple_node = mte_to_node(enode as *const maple_enode);
if mt_in_rcu(mt) {
mt_destroy_walk(enode, mt, false);
unsafe { call_rcu(core::ptr::addr_of_mut!((*node).__slate_anon_0.__slate_anon_1.rcu), Some(mt_free_walk)) };
} else {
mt_destroy_walk(enode, mt, true);
}
}

// Interface
/// mas_store() - Store an @entry.
/// @mas: The maple state.
/// @entry: The entry to store.
///
/// The @mas->index and @mas->last is used to set the range for the @entry.
///
/// Return: the first entry between mas->index and mas->last or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_store")]
extern "C" fn mas_store(mas: *mut ma_state, entry: *mut ()) -> *mut () {
let mut wr_mas: core::mem::MaybeUninit<ma_wr_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut ma_wr_state = core::ptr::addr_of_mut!(wr_mas) as *mut ma_wr_state;
unsafe {
core::ptr::write(__slate_slot_1, {
let mut __t0: ma_wr_state = core::mem::zeroed();
__t0.mas = mas;
__t0.entry = entry;
__t0.content = core::ptr::null_mut::<()>();
__t0.vacant_height = 0u8;
__t0.sufficient_height = 0u8;
__t0
});
mas_may_init_lock_check(mas);
trace_ma_write(___tp_str__4927, mas, 0u64, entry);
// Storing is the same operation as insert with the added caveat that it
// can overwrite entries.  Although this seems simple enough, one may
// want to examine what happens if a single store operation was to
// overwrite multiple entries within a self-balancing B-Tree.
mas_wr_prealloc_setup(core::ptr::addr_of_mut!(*__slate_slot_1));
(*mas).store_type = mas_wr_store_type(core::ptr::addr_of_mut!(*__slate_slot_1));
if ((*mas).mas_flags as i32) & 1 != 0 {
mas_wr_store_entry(core::ptr::addr_of_mut!(*__slate_slot_1));
'__join_7: {
loop {
if unlikely!(mas_is_err(mas)) != 0 {
break '__join_7;
} else {
if true {
break;
}
}
}
return (*__slate_slot_1).content;
}
core::arch::asm!("810: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 810b - ., 3; .popsection\n# {0}", const 810 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mas_store_2, sym __slate_asm_string_mas_store_3, const 4950 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
} else {
mas_prealloc_calc(core::ptr::addr_of_mut!(*__slate_slot_1), entry);
if (*mas).node_request != 0 {
mas_alloc_nodes(mas, (BIT!(11i32) as u32) | (BIT!(13i32) as u32));
if mas_is_err(mas) {
return core::ptr::null_mut::<()>();
}
}
mas_wr_store_entry(core::ptr::addr_of_mut!(*__slate_slot_1));
mas_destroy(mas);
return (*__slate_slot_1).content;
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__4927: *const u8 = (b"mas_store\0".as_ptr() as *mut u8) as *const u8;

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_store_811: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, *mut ()) -> *mut ()>, *mut ()>(Some(mas_store)) };

/// mas_store_gfp() - Store a value into the tree.
/// @mas: The maple state
/// @entry: The entry to store
/// @gfp: The GFP_FLAGS to use for allocations if necessary.
///
/// Return: 0 on success, -EINVAL on invalid request, -ENOMEM if memory could not
/// be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_store_gfp")]
extern "C" fn mas_store_gfp(mas: *mut ma_state, entry: *mut (), gfp: u32) -> i32 {
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut wr_mas: core::mem::MaybeUninit<ma_wr_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut ma_wr_state = core::ptr::addr_of_mut!(wr_mas) as *mut ma_wr_state;
let mut last: u64 = unsafe { core::mem::zeroed() };
let mut index: u64 = unsafe { core::mem::zeroed() };
unsafe {
index = (*mas).index;
last = (*mas).last;
core::ptr::write(__slate_slot_1, {
let mut __t0: ma_wr_state = core::mem::zeroed();
__t0.mas = mas;
__t0.entry = entry;
__t0.content = core::ptr::null_mut::<()>();
__t0.vacant_height = 0u8;
__t0.sufficient_height = 0u8;
__t0
});
ret = 0;
mas_may_init_lock_check(mas);
loop {
mas_wr_preallocate(core::ptr::addr_of_mut!(*__slate_slot_1), entry);
if unlikely!(mas_nomem(mas, gfp)) != 0 {
if entry.is_null() {
__mas_set_range(mas, index, last);
}
} else {
break;
}
}
if mas_is_err(mas) {
ret = xa_err((*mas).node as *mut ());
} else {
mas_wr_store_entry(core::ptr::addr_of_mut!(*__slate_slot_1));
}
mas_destroy(mas);
return ret;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_store_gfp_812: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, *mut (), u32) -> i32>, *mut ()>(Some(mas_store_gfp)) };

/// mas_store_prealloc() - Store a value into the tree using memory
/// preallocated in the maple state.
/// @mas: The maple state
/// @entry: The entry to store.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_store_prealloc")]
extern "C" fn mas_store_prealloc(mas: *mut ma_state, entry: *mut ()) {
let mut wr_mas: core::mem::MaybeUninit<ma_wr_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut ma_wr_state = core::ptr::addr_of_mut!(wr_mas) as *mut ma_wr_state;
unsafe {
'__slate_dispatch: {
core::ptr::write(__slate_slot_1, {
let mut __t0: ma_wr_state = core::mem::zeroed();
__t0.mas = mas;
__t0.entry = entry;
__t0.content = core::ptr::null_mut::<()>();
__t0.vacant_height = 0u8;
__t0.sufficient_height = 0u8;
__t0
});
mas_lock_check(mas);
if ((*mas).store_type as u32) == 2 {
mas_wr_prealloc_setup(core::ptr::addr_of_mut!(*__slate_slot_1));
} else {
mas_wr_walk_descend(core::ptr::addr_of_mut!(*__slate_slot_1));
if ((*mas).store_type as u32) != 4 {
// set wr_mas->content to current slot
(*__slate_slot_1).content = mas_slot_locked(mas, (*__slate_slot_1).slots, (*mas).offset);
mas_wr_end_piv(core::ptr::addr_of_mut!(*__slate_slot_1));
}
}
trace_ma_write(___tp_str__5030, mas, 0u64, entry);
mas_wr_store_entry(core::ptr::addr_of_mut!(*__slate_slot_1));
'__join_3: {
loop {
if unlikely!(mas_is_err(mas)) != 0 {
break '__join_3;
} else {
if true {
break;
}
}
}
mas_destroy(mas);
break '__slate_dispatch;
}
core::arch::asm!("813: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 813b - ., 3; .popsection\n# {0}", const 813 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mas_store_prealloc_2, sym __slate_asm_string_mas_store_prealloc_3, const 5032 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__5030: *const u8 = (b"mas_store_prealloc\0".as_ptr() as *mut u8) as *const u8;

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_store_prealloc_814: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, *mut ())>, *mut ()>(Some(mas_store_prealloc)) };

/// mas_preallocate() - Preallocate enough nodes for a store operation
/// @mas: The maple state
/// @entry: The entry that will be stored
/// @gfp: The GFP_FLAGS to use for allocations.
///
/// Return: 0 on success, -ENOMEM if memory could not be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_preallocate")]
extern "C" fn mas_preallocate(mas: *mut ma_state, entry: *mut (), gfp: u32) -> i32 {
let mut wr_mas: core::mem::MaybeUninit<ma_wr_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut ma_wr_state = core::ptr::addr_of_mut!(wr_mas) as *mut ma_wr_state;
unsafe {
core::ptr::write(__slate_slot_7, {
let mut __t0: ma_wr_state = core::mem::zeroed();
__t0.mas = mas;
__t0.entry = entry;
__t0.content = core::ptr::null_mut::<()>();
__t0.vacant_height = 0u8;
__t0.sufficient_height = 0u8;
__t0
});
mas_may_init_lock_check(mas);
mas_wr_prealloc_setup(core::ptr::addr_of_mut!(*__slate_slot_7));
(*mas).store_type = mas_wr_store_type(core::ptr::addr_of_mut!(*__slate_slot_7));
mas_prealloc_calc(core::ptr::addr_of_mut!(*__slate_slot_7), entry);
if (*mas).node_request != 0 {
let __v4: u8 = (((*mas).mas_flags as i32) & !1) as u8;
(*mas).mas_flags = __v4;
mas_alloc_nodes(mas, gfp);
if mas_is_err(mas) {
let ret: i32 = xa_err((*mas).node as *mut ());
(*mas).node_request = 0;
mas_destroy(mas);
mas_reset(mas);
return ret;
}
}
(*mas).mas_flags = (((*mas).mas_flags as i32) | 1) as u8;
return 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_preallocate_815: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, *mut (), u32) -> i32>, *mut ()>(Some(mas_preallocate)) };

/// mas_destroy() - destroy a maple state.
/// @mas: The maple state
///
/// Upon completion, check the left-most node and rebalance against the node to
/// the right if necessary.  Frees any allocated nodes associated with this maple
/// state.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_destroy")]
extern "C" fn mas_destroy(mas: *mut ma_state) {
let __v2: u8 = (((unsafe { (*mas).mas_flags }) as i32) & !1) as u8;
unsafe {
(*mas).mas_flags = __v2;
}
mas_empty_nodes(mas);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_destroy_816: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state)>, *mut ()>(Some(mas_destroy)) };

fn mas_may_activate(mas: *mut ma_state) {
if unsafe { (*mas).node }.is_null() {
unsafe {
(*mas).status = 1;
}
} else {
if (unsafe { (*mas).index }) > unsafe { (*mas).max } || (unsafe { (*mas).index }) < unsafe { (*mas).min } {
unsafe {
(*mas).status = 1;
}
} else {
unsafe {
(*mas).status = 0;
}
mas_lock_check(mas);
}
}
}

fn mas_next_setup(mas: *mut ma_state, max: u64, entry: &mut *mut ()) -> bool {
let was_none: bool = mas_is_none(mas as *const ma_state);
if unlikely!((unsafe { (*mas).last }) >= max) != 0 {
unsafe {
(*mas).status = 5;
}
return true;
}
// Retries on dead nodes handled by mas_walk
// Overflowed before, but the max changed
// The user expects the mas to be one before where it is
match (unsafe { (*mas).status }) as u32 {
0 => {
return false;
}
3 | 4 => {
unsafe {
(*mas).status = 1;
}
mas_walk(mas); // Retries on dead nodes handled by mas_walk
}
1 => {
mas_walk(mas); // Retries on dead nodes handled by mas_walk
}
5 => {
mas_may_activate(mas); // Overflowed before, but the max changed
}
6 => {
mas_may_activate(mas); // The user expects the mas to be one before where it is
unsafe {
*(entry as *mut *mut ()) = mas_walk(mas);
}
if !unsafe { *(entry as *mut *mut ()) }.is_null() {
return true;
}
}
2 => {
}
7 => {
return true;
}
_ => {
}
}
if likely!(mas_is_active(mas)) != 0 {
return false;
}
// Fast path
if mas_is_ptr(mas as *const ma_state) {
unsafe {
*(entry as *mut *mut ()) = core::ptr::null_mut::<()>();
}
if was_none && (unsafe { (*mas).index }) == 0 {
unsafe {
(*mas).last = 0;
}
unsafe {
(*mas).index = 0;
}
return true;
}
unsafe {
(*mas).index = 1;
}
unsafe {
(*mas).last = !0;
}
unsafe {
(*mas).status = 3;
}
return true;
}
if mas_is_none(mas as *const ma_state) {
return true;
}
false
}

/// mas_next() - Get the next entry.
/// @mas: The maple state
/// @max: The maximum index to check.
///
/// Returns the next entry after @mas->index.
/// Must hold rcu_read_lock or the write lock.
/// Can return the zero entry.
///
/// Return: The next entry or %NULL
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_next")]
extern "C" fn mas_next(mas: *mut ma_state, max: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
mas_may_init_lock_check(mas);
if mas_next_setup(mas, max, unsafe { &mut (*core::ptr::addr_of_mut!(entry)) }) {
return entry;
}
// Retries on dead nodes handled by mas_next_slot
mas_next_slot(mas, max, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_next_817: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64) -> *mut ()>, *mut ()>(Some(mas_next)) };

/// mas_next_range() - Advance the maple state to the next range
/// @mas: The maple state
/// @max: The maximum index to check.
///
/// Sets @mas->index and @mas->last to the range.
/// Must hold rcu_read_lock or the write lock.
/// Can return the zero entry.
///
/// Return: The next entry or %NULL
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_next_range")]
extern "C" fn mas_next_range(mas: *mut ma_state, max: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
mas_may_init_lock_check(mas);
if mas_next_setup(mas, max, unsafe { &mut (*core::ptr::addr_of_mut!(entry)) }) {
return entry;
}
// Retries on dead nodes handled by mas_next_slot
mas_next_slot(mas, max, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_next_range_818: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64) -> *mut ()>, *mut ()>(Some(mas_next_range)) };

/// mt_next() - get the next value in the maple tree
/// @mt: The maple tree
/// @index: The start index
/// @max: The maximum index to check
///
/// Takes RCU read lock internally to protect the search, which does not
/// protect the returned pointer after dropping RCU read lock.
/// See also: Documentation/core-api/maple_tree.rst
///
/// Return: The entry higher than @index or %NULL if nothing is found.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mt_next")]
extern "C" fn mt_next(mt: *mut maple_tree, index: u64, max: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
let mut mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = mt;
__t0.index = index;
__t0.last = index;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
rcu_read_lock();
entry = mas_next(core::ptr::addr_of_mut!(mas), max);
rcu_read_unlock();
entry
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mt_next_819: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, u64, u64) -> *mut ()>, *mut ()>(Some(mt_next)) };

fn mas_prev_setup(mas: *mut ma_state, min: u64, entry: &mut *mut ()) -> bool {
if unlikely!((unsafe { (*mas).index }) <= min) != 0 {
unsafe {
(*mas).status = 6;
}
return true;
}
match (unsafe { (*mas).status }) as u32 {
0 => {
return false;
}
1 => {
}
3 | 4 => {
unsafe {
(*mas).status = 1;
}
}
6 => {
mas_may_activate(mas); // underflowed before but the min changed
}
5 => {
mas_may_activate(mas); // User expects mas to be one after where it is
unsafe {
*(entry as *mut *mut ()) = mas_walk(mas);
}
if !unsafe { *(entry as *mut *mut ()) }.is_null() {
return true;
}
}
2 => {
}
7 => {
return true;
}
_ => {
}
}
if mas_is_start(mas as *const ma_state) {
mas_walk(mas);
}
if unlikely!(mas_is_ptr(mas as *const ma_state)) != 0 {
if (unsafe { (*mas).index }) == 0 {
unsafe {
(*mas).status = 3;
}
return true;
}
unsafe {
(*mas).last = 0;
}
unsafe {
(*mas).index = 0;
}
unsafe {
*(entry as *mut *mut ()) = mas_root(mas);
}
return true;
}
if mas_is_none(mas as *const ma_state) {
if (unsafe { (*mas).index }) != 0 {
// Walked to out-of-range pointer?
unsafe {
(*mas).last = 0;
}
unsafe {
(*mas).index = 0;
}
unsafe {
(*mas).status = 2;
}
unsafe {
*(entry as *mut *mut ()) = mas_root(mas);
}
return true;
}
return true;
}
false
}

/// mas_prev() - Get the previous entry
/// @mas: The maple state
/// @min: The minimum value to check.
///
/// Must hold rcu_read_lock or the write lock.
/// Will reset mas to ma_start if the status is ma_none.  Will stop on not
/// searchable nodes.
///
/// Return: the previous value or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_prev")]
extern "C" fn mas_prev(mas: *mut ma_state, min: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
mas_may_init_lock_check(mas);
if mas_prev_setup(mas, min, unsafe { &mut (*core::ptr::addr_of_mut!(entry)) }) {
return entry;
}
mas_prev_slot(mas, min, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_prev_820: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64) -> *mut ()>, *mut ()>(Some(mas_prev)) };

/// mas_prev_range() - Advance to the previous range
/// @mas: The maple state
/// @min: The minimum value to check.
///
/// Sets @mas->index and @mas->last to the range.
/// Must hold rcu_read_lock or the write lock.
/// Will reset mas to ma_start if the node is ma_none.  Will stop on not
/// searchable nodes.
///
/// Return: the previous value or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_prev_range")]
extern "C" fn mas_prev_range(mas: *mut ma_state, min: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
mas_may_init_lock_check(mas);
if mas_prev_setup(mas, min, unsafe { &mut (*core::ptr::addr_of_mut!(entry)) }) {
return entry;
}
mas_prev_slot(mas, min, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_prev_range_821: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64) -> *mut ()>, *mut ()>(Some(mas_prev_range)) };

/// mt_prev() - get the previous value in the maple tree
/// @mt: The maple tree
/// @index: The start index
/// @min: The minimum index to check
///
/// Takes RCU read lock internally to protect the search, which does not
/// protect the returned pointer after dropping RCU read lock.
/// See also: Documentation/core-api/maple_tree.rst
///
/// Return: The entry before @index or %NULL if nothing is found.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mt_prev")]
extern "C" fn mt_prev(mt: *mut maple_tree, index: u64, min: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
let mut mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = mt;
__t0.index = index;
__t0.last = index;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
rcu_read_lock();
entry = mas_prev(core::ptr::addr_of_mut!(mas), min);
rcu_read_unlock();
entry
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mt_prev_822: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, u64, u64) -> *mut ()>, *mut ()>(Some(mt_prev)) };

/// mas_pause() - Pause a mas_find/mas_for_each to drop the lock.
/// @mas: The maple state to pause
///
/// Some users need to pause a walk and drop the lock they're holding in
/// order to yield to a higher priority thread or carry out an operation
/// on an entry.  Those users should call this function before they drop
/// the lock.  It resets the @mas to be suitable for the next iteration
/// of the loop after the user has reacquired the lock.  If most entries
/// found during a walk require you to call mas_pause(), the mt_for_each()
/// iterator may be more appropriate.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_pause")]
extern "C" fn mas_pause(mas: *mut ma_state) {
mas_lock_check(mas);
unsafe {
(*mas).status = 4;
}
unsafe {
(*mas).node = core::ptr::null_mut::<maple_enode>();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_pause_823: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state)>, *mut ()>(Some(mas_pause)) };

/// mas_find_setup() - Internal function to set up mas_find*().
/// @mas: The maple state
/// @max: The maximum index
/// @entry: Pointer to the entry
///
/// Returns: True if entry is the answer, false otherwise.
#[inline(always)]
fn mas_find_setup(mas: *mut ma_state, max: u64, entry: &mut *mut ()) -> bool {
unsafe {
let __t0: u32 = (*mas).status as u32;
if __t0 == 0 {
if (*mas).last < max {
return false;
} else {
return true;
}
} else {
if __t0 != 1 {
if __t0 == 4 {
if unlikely!((*mas).last >= max) != 0 {
return true;
} else {
let __v1: u64 = (*mas).last;
let __v0: u64 = __v1.wrapping_add(1);
(*mas).last = __v0;
(*mas).index = __v0;
(*mas).status = 1;
}
} else {
if __t0 == 3 {
if unlikely!((*mas).last >= max) != 0 {
return true;
} else {
(*mas).index = (*mas).last;
(*mas).status = 1;
}
} else {
if __t0 == 6 {
if unlikely!((*mas).index >= max) != 0 {
(*mas).status = 5;
return true;
} else {
// mas is pointing at entry before unable to go lower
mas_may_activate(mas);
*(entry as *mut *mut ()) = mas_walk(mas);
if !unsafe { *(entry as *mut *mut ()) }.is_null() {
return true;
}
}
} else {
if __t0 == 5 {
if unlikely!((*mas).last >= max) != 0 {
return true;
} else {
mas_may_activate(mas);
*(entry as *mut *mut ()) = mas_walk(mas);
if !unsafe { *(entry as *mut *mut ()) }.is_null() {
return true;
}
}
} else {
if __t0 != 2 {
if __t0 == 7 {
return true;
}
}
}
}
}
}
}
}
if mas_is_start(mas as *const ma_state) {
// First run or continue
if (*mas).index > max {
return true;
} else {
*(entry as *mut *mut ()) = mas_walk(mas);
if !unsafe { *(entry as *mut *mut ()) }.is_null() {
return true;
}
}
}
if unlikely!(mas_is_ptr(mas as *const ma_state)) != 0 {
(*mas).status = 3;
(*mas).index = 1;
(*mas).last = !0;
return true;
} else {
if unlikely!(mas_is_none(mas as *const ma_state)) != 0 {
return true;
} else {
if (*mas).index == max {
return true;
} else {
return false;
}
}
}
}
}

/// mas_find() - On the first call, find the entry at or after mas->index up to
/// %max.  Otherwise, find the entry after mas->index.
/// @mas: The maple state
/// @max: The maximum value to check.
///
/// Must hold rcu_read_lock or the write lock.
/// If an entry exists, last and index are updated accordingly.
/// May set @mas->status to ma_overflow.
///
/// Return: The entry or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_find")]
extern "C" fn mas_find(mas: *mut ma_state, max: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
mas_may_init_lock_check(mas);
if mas_find_setup(mas, max, unsafe { &mut (*core::ptr::addr_of_mut!(entry)) }) {
return entry;
}
// Retries on dead nodes handled by mas_next_slot
entry = mas_next_slot(mas, max, false);
// Ignore overflow
unsafe {
(*mas).status = 0;
}
entry
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_find_824: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64) -> *mut ()>, *mut ()>(Some(mas_find)) };

/// mas_find_range() - On the first call, find the entry at or after
/// mas->index up to %max.  Otherwise, advance to the next slot mas->index.
/// @mas: The maple state
/// @max: The maximum value to check.
///
/// Must hold rcu_read_lock or the write lock.
/// If an entry exists, last and index are updated accordingly.
/// May set @mas->status to ma_overflow.
///
/// Return: The entry or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_find_range")]
extern "C" fn mas_find_range(mas: *mut ma_state, max: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
mas_may_init_lock_check(mas);
if mas_find_setup(mas, max, unsafe { &mut (*core::ptr::addr_of_mut!(entry)) }) {
return entry;
}
// Retries on dead nodes handled by mas_next_slot
mas_next_slot(mas, max, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_find_range_825: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64) -> *mut ()>, *mut ()>(Some(mas_find_range)) };

/// mas_find_rev_setup() - Internal function to set up mas_find_*_rev()
/// @mas: The maple state
/// @min: The minimum index
/// @entry: Pointer to the entry
///
/// Returns: True if entry is the answer, false otherwise.
fn mas_find_rev_setup(mas: *mut ma_state, min: u64, entry: &mut *mut ()) -> bool {
unsafe {
'__join_3: {
'__join_0: {
let __t0: u32 = (*mas).status as u32;
if __t0 == 0 {
break '__join_3;
} else {
if __t0 != 1 {
if __t0 == 4 {
if unlikely!((*mas).index <= min) != 0 {
(*mas).status = 6;
return true;
} else {
let __v1: u64 = (*mas).index;
let __v0: u64 = __v1.wrapping_sub(1);
(*mas).index = __v0;
(*mas).last = __v0;
(*mas).status = 1;
}
} else {
if __t0 == 3 {
if (*mas).index <= min {
break '__join_0;
} else {
(*mas).last = (*mas).index;
(*mas).status = 1;
}
} else {
if __t0 == 5 {
if unlikely!((*mas).index <= min) != 0 {
(*mas).status = 6;
return true;
} else {
// user expects the mas to be one after where it is
(*mas).status = 0;
}
} else {
if __t0 == 6 {
if unlikely!((*mas).index <= min) != 0 {
return true;
} else {
// user expects the mas to be one before where it is
(*mas).status = 0;
}
} else {
if __t0 != 2 {
if __t0 == 7 {
return true;
}
}
}
}
}
}
}
}
if mas_is_start(mas as *const ma_state) {
// First run or continue
if (*mas).index < min {
return true;
} else {
*(entry as *mut *mut ()) = mas_walk(mas);
if !unsafe { *(entry as *mut *mut ()) }.is_null() {
return true;
}
}
}
if unlikely!(mas_is_ptr(mas as *const ma_state)) == 0 {
if unlikely!(mas_is_none(mas as *const ma_state)) != 0 {
// Walked to the location, and there was nothing so the previous
// location is 0.
(*mas).index = 0;
(*mas).last = 0;
(*mas).status = 2;
*(entry as *mut *mut ()) = mas_root(mas);
return true;
} else {
break '__join_3;
}
}
}
(*mas).status = 3;
return true;
}
if (*mas).index < min {
return true;
} else {
return false;
}
}
}

/// mas_find_rev: On the first call, find the first non-null entry at or below
/// mas->index down to %min.  Otherwise find the first non-null entry below
/// mas->index down to %min.
/// @mas: The maple state
/// @min: The minimum value to check.
///
/// Must hold rcu_read_lock or the write lock.
/// If an entry exists, last and index are updated accordingly.
/// May set @mas->status to ma_underflow.
///
/// Return: The entry or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_find_rev")]
extern "C" fn mas_find_rev(mas: *mut ma_state, min: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
mas_may_init_lock_check(mas);
if mas_find_rev_setup(mas, min, unsafe { &mut (*core::ptr::addr_of_mut!(entry)) }) {
return entry;
}
// Retries on dead nodes handled by mas_prev_slot
mas_prev_slot(mas, min, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_find_rev_826: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64) -> *mut ()>, *mut ()>(Some(mas_find_rev)) };

/// mas_find_range_rev: On the first call, find the first non-null entry at or
/// below mas->index down to %min.  Otherwise advance to the previous slot after
/// mas->index down to %min.
/// @mas: The maple state
/// @min: The minimum value to check.
///
/// Must hold rcu_read_lock or the write lock.
/// If an entry exists, last and index are updated accordingly.
/// May set @mas->status to ma_underflow.
///
/// Return: The entry or %NULL.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_find_range_rev")]
extern "C" fn mas_find_range_rev(mas: *mut ma_state, min: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
mas_may_init_lock_check(mas);
if mas_find_rev_setup(mas, min, unsafe { &mut (*core::ptr::addr_of_mut!(entry)) }) {
return entry;
}
// Retries on dead nodes handled by mas_prev_slot
mas_prev_slot(mas, min, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_find_range_rev_827: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state, u64) -> *mut ()>, *mut ()>(Some(mas_find_range_rev)) };

/// mas_erase() - Find the range in which index resides and erase the entire
/// range.
/// @mas: The maple state
///
/// Must hold the write lock.
/// Searches for @mas->index, sets @mas->index and @mas->last to the range and
/// erases that range.
///
/// Note that erase requires allocations and will use GFP_KERNEL | __GFP_NOFAIL
/// to do so if necessary.  If the allocation fails, the internal lock will be
/// dropped to retry.  An externally locked tree must be protected by a lock that
/// allows blocking for this API.
///
/// Return: the entry that was erased or %NULL, @mas->index and @mas->last are updated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mas_erase")]
extern "C" fn mas_erase(mas: *mut ma_state) -> *mut () {
let mut wr_mas: core::mem::MaybeUninit<ma_wr_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut ma_wr_state = core::ptr::addr_of_mut!(wr_mas) as *mut ma_wr_state;
let mut index: u64 = unsafe { core::mem::zeroed() };
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
index = (*mas).index;
core::ptr::write(__slate_slot_0, {
let mut __t0: ma_wr_state = core::mem::zeroed();
__t0.mas = mas;
__t0.entry = core::ptr::null_mut::<()>();
__t0.content = core::ptr::null_mut::<()>();
__t0.vacant_height = 0u8;
__t0.sufficient_height = 0u8;
__t0
});
// In low memory situations, the allocation is retried with the gfp flag
// GFP_KERNEL | __GFP_NOFAIL.  The internal spinlock is dropped in
// mas_nomem_nofail(), however the external lock is not dropped.
if mt_external_lock((*mas).tree as *const maple_tree) {
might_alloc(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
mas_make_walkable(mas);
'__join_0: {
loop {
*__slate_slot_2 = mas_state_walk(mas);
if (*__slate_slot_2).is_null() {
break '__join_0;
} else {
// Must reset to ensure spanning writes of last slot are detected
mas_reset(mas);
mas_wr_preallocate(core::ptr::addr_of_mut!(*__slate_slot_0), core::ptr::null_mut::<()>());
if !mas_nomem_nofail(mas, index, index) {
break;
}
}
}
mas_wr_store_entry(core::ptr::addr_of_mut!(*__slate_slot_0));
}
mas_destroy(mas);
return *__slate_slot_2;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mas_erase_828: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ma_state) -> *mut ()>, *mut ()>(Some(mas_erase)) };

/// mas_nomem() - Check if there was an error allocating and do the allocation
/// if necessary.
///
/// @mas: The maple state
/// @gfp: The GFP_FLAGS to use for allocations
/// Return: False on no memory.  True otherwise (partial success as well)
#[unsafe(no_mangle)]
extern "C" fn mas_nomem(mas: *mut ma_state, gfp: u32) -> bool {
if likely!((unsafe { (*mas).node }) != MA_ERROR!(12i32.wrapping_neg())) != 0 {
return false;
}
// Allocations can fail, don't do this.
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = !gfpflags_allow_blocking(gfp) && mt_external_lock((unsafe { (*mas).tree }) as *const maple_tree);
__ret_warn_on = __v1 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("829: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 829b - ., 3; .popsection\n# {0}", const 829 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mas_nomem_2, sym __slate_asm_string_mas_nomem_3, const 5728 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("830: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 830b - ., 4; .popsection\n# {0}", const 830 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
let __v6: bool = gfpflags_allow_blocking(gfp) && !mt_external_lock((unsafe { (*mas).tree }) as *const maple_tree);
if __v6 {
mtree_unlock!(unsafe { (*mas).tree });
mas_alloc_nodes(mas, gfp);
mtree_lock!(unsafe { (*mas).tree });
} else {
mas_alloc_nodes(mas, gfp);
}
// Return false on zero forward progress.  Partial allocations are kept
// so the retry path will attempt to get the rest.  The failure should
// not happen as we try our best to reclaim.  The user would need an
// external lock with a non-blocking gfp in a low memory situation.
if unsafe { (*mas).sheaf }.is_null() && unsafe { (*mas).alloc }.is_null() {
return false;
}
mas_reset(mas);
true
}

/// mas_nomem_nofail() - Retry allocations with __GFP_NOFAIL, if the maple state
/// has stored the -ENOMEM error.
/// @mas: The maple state
/// @index: The start of the range for the @mas reset
/// @last: The end of the range for the @mas reset
///
/// Return: false if @mas isn't in an -ENOMEM state.  True if the allocation
/// happens, the state is reset.  The internal lock will be dropped and external
/// locks must allow blocking.
#[unsafe(no_mangle)]
extern "C" fn mas_nomem_nofail(mas: *mut ma_state, index: u64, last: u64) -> bool {
let mut gfp: u32 = 0;
if likely!((unsafe { (*mas).node }) != MA_ERROR!(12i32.wrapping_neg())) != 0 {
return false;
}
gfp = ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32) | (BIT!(15i32) as u32);
if !mt_external_lock((unsafe { (*mas).tree }) as *const maple_tree) {
mtree_unlock!(unsafe { (*mas).tree });
mas_alloc_nodes(mas, gfp);
mtree_lock!(unsafe { (*mas).tree });
} else {
mas_alloc_nodes(mas, gfp);
}
mas_set_range(mas, index, last);
true
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn maple_tree_init() {
let mut args: kmem_cache_args = {
let mut __t0: kmem_cache_args = unsafe { core::mem::zeroed() };
__t0.align = core::mem::size_of::<maple_node>() as u32;
__t0.sheaf_capacity = 32u32;
__t0
};
unsafe {
maple_node_cache = __kmem_cache_create_args(c"maple_node".as_ptr() as *const u8, core::mem::size_of::<maple_node>() as u32, core::ptr::addr_of_mut!(args), __SLAB_FLAG_BIT!(8i32));
}
}

/// mtree_load() - Load a value stored in a maple tree
/// @mt: The maple tree
/// @index: The index to load
///
/// Return: the entry or %NULL
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_load")]
extern "C" fn mtree_load(mt: *mut maple_tree, index: u64) -> *mut () {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
let mut mas: core::mem::MaybeUninit<ma_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut ma_state = core::ptr::addr_of_mut!(mas) as *mut ma_state;
unsafe {
core::ptr::write(__slate_slot_3, {
let mut __t0: ma_state = core::mem::zeroed();
__t0.tree = mt;
__t0.index = index;
__t0.last = index;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
});
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
trace_ma_read(___tp_str__5808, core::ptr::addr_of_mut!(*__slate_slot_3));
rcu_read_lock();
'__join_2: {
loop {
*__slate_slot_2 = mas_start(core::ptr::addr_of_mut!(*__slate_slot_3)) as *mut ();
if unlikely!(mas_is_none(core::ptr::addr_of_mut!(*__slate_slot_3) as *const ma_state)) != 0 {
break '__join_2;
} else {
if unlikely!(mas_is_ptr(core::ptr::addr_of_mut!(*__slate_slot_3) as *const ma_state)) != 0 {
break;
} else {
*__slate_slot_2 = mtree_lookup_walk(core::ptr::addr_of_mut!(*__slate_slot_3));
if (*__slate_slot_2).is_null() {
__v0 = unlikely!(mas_is_start(core::ptr::addr_of_mut!(*__slate_slot_3) as *const ma_state)) != 0;
} else {
__v0 = false;
}
if !__v0 {
break '__join_2;
}
}
}
}
if index != 0 {
*__slate_slot_2 = core::ptr::null_mut::<()>();
}
}
rcu_read_unlock();
if xa_is_zero(*__slate_slot_2 as *const ()) {
return core::ptr::null_mut::<()>();
} else {
return *__slate_slot_2;
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__5808: *const u8 = (b"mtree_load\0".as_ptr() as *mut u8) as *const u8;

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_load_831: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, u64) -> *mut ()>, *mut ()>(Some(mtree_load)) };

/// mtree_store_range() - Store an entry at a given range.
/// @mt: The maple tree
/// @index: The start of the range
/// @last: The end of the range
/// @entry: The entry to store
/// @gfp: The GFP_FLAGS to use for allocations
///
/// Return: 0 on success, -EINVAL on invalid request, -ENOMEM if memory could not
/// be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_store_range")]
extern "C" fn mtree_store_range(mt: *mut maple_tree, index: u64, last: u64, entry: *mut (), gfp: u32) -> i32 {
let mut mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = mt;
__t0.index = index;
__t0.last = last;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
let mut ret: i32 = 0;
let mut __v0: *const u8;
__v0 = unsafe { ___tp_str__5851 };
trace_ma_write(__v0, core::ptr::addr_of_mut!(mas), 0u64, entry);
let mut __v1: i64;
let __ret_warn_on: i32 = !(!xa_is_advanced(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("832: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 832b - ., 3; .popsection\n# {0}", const 832 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mtree_store_range_2, sym __slate_asm_string_mtree_store_range_3, const 5852 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("833: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 833b - ., 4; .popsection\n# {0}", const 833 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
if __v1 != 0 {
return 22i32.wrapping_neg();
}
if index > last {
return 22i32.wrapping_neg();
}
mtree_lock!(mt);
ret = mas_store_gfp(core::ptr::addr_of_mut!(mas), entry, gfp);
mtree_unlock!(mt);
ret
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__5851: *const u8 = (b"mtree_store_range\0".as_ptr() as *mut u8) as *const u8;

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_store_range_834: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, u64, u64, *mut (), u32) -> i32>, *mut ()>(Some(mtree_store_range)) };

/// mtree_store() - Store an entry at a given index.
/// @mt: The maple tree
/// @index: The index to store the value
/// @entry: The entry to store
/// @gfp: The GFP_FLAGS to use for allocations
///
/// Return: 0 on success, -EINVAL on invalid request, -ENOMEM if memory could not
/// be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_store")]
extern "C" fn mtree_store(mt: *mut maple_tree, index: u64, entry: *mut (), gfp: u32) -> i32 {
mtree_store_range(mt, index, index, entry, gfp)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_store_835: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, u64, *mut (), u32) -> i32>, *mut ()>(Some(mtree_store)) };

/// mtree_insert_range() - Insert an entry from [first, last] at a given range
///                        if there is no value.
/// @mt: The maple tree
/// @first: The start of the range
/// @last: The end of the range (inclusive)
/// @entry: The entry to store
/// @gfp: The GFP_FLAGS to use for allocations.
///
/// Note that @last is inclusive. That is, @last = @first + length - 1;
///
/// Return: 0 on success, -EEXISTS if the range is occupied, -EINVAL on invalid
/// request, -ENOMEM if memory could not be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_insert_range")]
extern "C" fn mtree_insert_range(mt: *mut maple_tree, first: u64, last: u64, entry: *mut (), gfp: u32) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut ms: core::mem::MaybeUninit<ma_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut ma_state = core::ptr::addr_of_mut!(ms) as *mut ma_state;
unsafe {
core::ptr::write(__slate_slot_4, {
let mut __t0: ma_state = core::mem::zeroed();
__t0.tree = mt;
__t0.index = first;
__t0.last = last;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
});
ret = 0;
__ret_warn_on = !(!xa_is_advanced(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("836: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 836b - ., 3; .popsection\n# {0}", const 836 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mtree_insert_range_5, sym __slate_asm_string_mtree_insert_range_6, const 5903 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("837: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 837b - ., 4; .popsection\n# {0}", const 837 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 22i32.wrapping_neg();
} else {
if first > last {
return 22i32.wrapping_neg();
} else {
mtree_lock!(mt);
loop {
mas_insert(core::ptr::addr_of_mut!(*__slate_slot_4), entry);
if !mas_nomem(core::ptr::addr_of_mut!(*__slate_slot_4), gfp) {
break;
}
}
mtree_unlock!(mt);
if mas_is_err(core::ptr::addr_of_mut!(*__slate_slot_4)) {
ret = xa_err((*__slate_slot_4).node as *mut ());
}
mas_destroy(core::ptr::addr_of_mut!(*__slate_slot_4));
return ret;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_insert_range_838: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, u64, u64, *mut (), u32) -> i32>, *mut ()>(Some(mtree_insert_range)) };

/// mtree_insert() - Insert an entry at a given index if there is no value.
/// @mt: The maple tree
/// @index : The index to store the value
/// @entry: The entry to store
/// @gfp: The GFP_FLAGS to use for allocations.
///
/// Return: 0 on success, -EEXISTS if the range is occupied, -EINVAL on invalid
/// request, -ENOMEM if memory could not be allocated.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_insert")]
extern "C" fn mtree_insert(mt: *mut maple_tree, index: u64, entry: *mut (), gfp: u32) -> i32 {
mtree_insert_range(mt, index, index, entry, gfp)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_insert_839: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, u64, *mut (), u32) -> i32>, *mut ()>(Some(mtree_insert)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_alloc_range")]
extern "C" fn mtree_alloc_range(mt: *mut maple_tree, startp: *mut u64, entry: *mut (), size: u64, min: u64, max: u64, gfp: u32) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut mas: core::mem::MaybeUninit<ma_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut ma_state = core::ptr::addr_of_mut!(mas) as *mut ma_state;
let mut ret: i32 = unsafe { core::mem::zeroed() };
unsafe {
ret = 0;
core::ptr::write(__slate_slot_3, {
let mut __t0: ma_state = core::mem::zeroed();
__t0.tree = mt;
__t0.index = 0u64;
__t0.last = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
});
if !mt_is_alloc(mt) {
return 22i32.wrapping_neg();
} else {
__ret_warn_on = !(!mt_is_reserved(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("840: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 840b - ., 3; .popsection\n# {0}", const 840 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mtree_alloc_range_5, sym __slate_asm_string_mtree_alloc_range_6, const 5951 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("841: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 841b - ., 4; .popsection\n# {0}", const 841 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 22i32.wrapping_neg();
} else {
mtree_lock!(mt);
'__join_0: {
loop {
ret = mas_empty_area(core::ptr::addr_of_mut!(*__slate_slot_3), min, max, size);
if ret != 0 {
break '__join_0;
} else {
mas_insert(core::ptr::addr_of_mut!(*__slate_slot_3), entry);
// mas_nomem() may release the lock, causing the allocated area
// to be unavailable, so try to allocate a free area again.
if !mas_nomem(core::ptr::addr_of_mut!(*__slate_slot_3), gfp) {
break;
}
}
}
if mas_is_err(core::ptr::addr_of_mut!(*__slate_slot_3)) {
ret = xa_err((*__slate_slot_3).node as *mut ());
} else {
*startp = (*__slate_slot_3).index;
}
}
mtree_unlock!(mt);
mas_destroy(core::ptr::addr_of_mut!(*__slate_slot_3));
return ret;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_alloc_range_842: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, *mut u64, *mut (), u64, u64, u64, u32) -> i32>, *mut ()>(Some(mtree_alloc_range)) };

/// mtree_alloc_cyclic() - Find somewhere to store this entry in the tree.
/// @mt: The maple tree.
/// @startp: Pointer to ID.
/// @range_lo: Lower bound of range to search.
/// @range_hi: Upper bound of range to search.
/// @entry: The entry to store.
/// @next: Pointer to next ID to allocate.
/// @gfp: The GFP_FLAGS to use for allocations.
///
/// Finds an empty entry in @mt after @next, stores the new index into
/// the @id pointer, stores the entry at that index, then updates @next.
///
/// @mt must be initialized with the MT_FLAGS_ALLOC_RANGE flag.
///
/// Context: Any context.  Takes and releases the mt.lock.  May sleep if
/// the @gfp flags permit.
///
/// Return: 0 if the allocation succeeded without wrapping, 1 if the
/// allocation succeeded after wrapping, -ENOMEM if memory could not be
/// allocated, -EINVAL if @mt cannot be used, or -EBUSY if there are no
/// free entries.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_alloc_cyclic")]
extern "C" fn mtree_alloc_cyclic(mt: *mut maple_tree, startp: *mut u64, entry: *mut (), range_lo: u64, range_hi: u64, next: *mut u64, gfp: u32) -> i32 {
let mut ret: i32 = 0;
let mut mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = mt;
__t0.index = 0u64;
__t0.last = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
if !mt_is_alloc(mt) {
return 22i32.wrapping_neg();
}
let mut __v0: i64;
let __ret_warn_on: i32 = !(!mt_is_reserved(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("843: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 843b - ., 3; .popsection\n# {0}", const 843 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mtree_alloc_cyclic_1, sym __slate_asm_string_mtree_alloc_cyclic_2, const 6013 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("844: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 844b - ., 4; .popsection\n# {0}", const 844 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 22i32.wrapping_neg();
}
mtree_lock!(mt);
ret = mas_alloc_cyclic(core::ptr::addr_of_mut!(mas), startp, entry, range_lo, range_hi, next, gfp);
mtree_unlock!(mt);
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_alloc_cyclic_845: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, *mut u64, *mut (), u64, u64, *mut u64, u32) -> i32>, *mut ()>(Some(mtree_alloc_cyclic)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_alloc_rrange")]
extern "C" fn mtree_alloc_rrange(mt: *mut maple_tree, startp: *mut u64, entry: *mut (), size: u64, min: u64, max: u64, gfp: u32) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut mas: core::mem::MaybeUninit<ma_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut ma_state = core::ptr::addr_of_mut!(mas) as *mut ma_state;
let mut ret: i32 = unsafe { core::mem::zeroed() };
unsafe {
ret = 0;
core::ptr::write(__slate_slot_3, {
let mut __t0: ma_state = core::mem::zeroed();
__t0.tree = mt;
__t0.index = 0u64;
__t0.last = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
});
if !mt_is_alloc(mt) {
return 22i32.wrapping_neg();
} else {
__ret_warn_on = !(!mt_is_reserved(entry as *const ())) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("846: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 846b - ., 3; .popsection\n# {0}", const 846 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mtree_alloc_rrange_5, sym __slate_asm_string_mtree_alloc_rrange_6, const 6033 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("847: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 847b - ., 4; .popsection\n# {0}", const 847 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
return 22i32.wrapping_neg();
} else {
mtree_lock!(mt);
'__join_0: {
loop {
ret = mas_empty_area_rev(core::ptr::addr_of_mut!(*__slate_slot_3), min, max, size);
if ret != 0 {
break '__join_0;
} else {
mas_insert(core::ptr::addr_of_mut!(*__slate_slot_3), entry);
// mas_nomem() may release the lock, causing the allocated area
// to be unavailable, so try to allocate a free area again.
if !mas_nomem(core::ptr::addr_of_mut!(*__slate_slot_3), gfp) {
break;
}
}
}
if mas_is_err(core::ptr::addr_of_mut!(*__slate_slot_3)) {
ret = xa_err((*__slate_slot_3).node as *mut ());
} else {
*startp = (*__slate_slot_3).index;
}
}
mtree_unlock!(mt);
mas_destroy(core::ptr::addr_of_mut!(*__slate_slot_3));
return ret;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_alloc_rrange_848: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, *mut u64, *mut (), u64, u64, u64, u32) -> i32>, *mut ()>(Some(mtree_alloc_rrange)) };

/// mtree_erase() - Find an index and erase the entire range.
/// @mt: The maple tree
/// @index: The index to erase
///
/// Erasing is the same as a walk to an entry then a store of a NULL to that
/// ENTIRE range.  In fact, it is implemented as such using the advanced API.
///
/// Note that erase requires allocations and will use GFP_KERNEL | __GFP_NOFAIL
/// to do so if necessary.  If the allocation fails, the internal lock will be
/// dropped to retry.
///
/// Return: The entry stored at the @index or %NULL
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_erase")]
extern "C" fn mtree_erase(mt: *mut maple_tree, index: u64) -> *mut () {
let mut entry: *mut () = core::ptr::null_mut::<()>();
let mut mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = mt;
__t0.index = index;
__t0.last = index;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
let mut __v0: *const u8;
__v0 = unsafe { ___tp_str__6081 };
trace_ma_op(__v0, core::ptr::addr_of_mut!(mas));
might_alloc(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
mtree_lock!(mt);
entry = mas_erase(core::ptr::addr_of_mut!(mas));
mtree_unlock!(mt);
entry
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__6081: *const u8 = (b"mtree_erase\0".as_ptr() as *mut u8) as *const u8;

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_erase_849: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, u64) -> *mut ()>, *mut ()>(Some(mtree_erase)) };

/// mas_dup_free() - Free an incomplete duplication of a tree.
/// @mas: The maple state of a incomplete tree.
///
/// The parameter @mas->node passed in indicates that the allocation failed on
/// this node. This function frees all nodes starting from @mas->node in the
/// reverse order of mas_dup_build(). There is no need to hold the source tree
/// lock at this time.
fn mas_dup_free(mas: *mut ma_state) {
let mut node: *mut maple_node = unsafe { core::mem::zeroed() };
let mut r#type: u32 = 0;
let mut count: u8 = 0;
let mut i: u8 = 0;
// Maybe the first node allocation failed.
if mas_is_none(mas as *const ma_state) {
return;
}
while !mte_is_root((unsafe { (*mas).node }) as *const maple_enode) {
mas_ascend(mas);
if (unsafe { (*mas).offset }) != 0 {
unsafe {
(*mas).offset = ((*mas).offset as i32).wrapping_sub(1) as u8;
}
loop {
mas_descend(mas);
unsafe {
(*mas).offset = mas_data_end(mas);
}
if !(!mte_is_leaf((unsafe { (*mas).node }) as *const maple_enode)) {
break;
}
}
mas_ascend(mas);
}
node = mte_to_node((unsafe { (*mas).node }) as *const maple_enode);
r#type = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
let slots: *mut *mut () = ma_slots(node, r#type);
count = (mas_data_end(mas) as i32).wrapping_add(1) as u8;
i = 0;
while (i as i32) < (count as i32) {
let __v4: *mut u64 = (slots as *mut u64).wrapping_offset((i as i32) as isize);
let __v6: u64 = (unsafe { *__v4 }) & !255;
unsafe {
*__v4 = __v6;
}
i = (i as i32).wrapping_add(1) as u8;
}
mt_free_bulk(count as u64, slots);
}
node = mte_to_node((unsafe { (*mas).node }) as *const maple_enode);
unsafe { kfree(node as *const ()) };
}

/// mas_copy_node() - Copy a maple node and replace the parent.
/// @mas: The maple state of source tree.
/// @new_mas: The maple state of new tree.
/// @parent: The parent of the new node.
///
/// Copy @mas->node to @new_mas->node, set @parent to be the parent of
/// @new_mas->node. If memory allocation fails, @mas is set to -ENOMEM.
fn mas_copy_node(mas: *mut ma_state, new_mas: *mut ma_state, parent: *mut maple_pnode) {
let node: *mut maple_node = mte_to_node((unsafe { (*mas).node }) as *const maple_enode);
let new_node: *mut maple_node = mte_to_node((unsafe { (*new_mas).node }) as *const maple_enode);
let mut val: u64 = 0;
// Copy the node completely.
unsafe { memcpy(new_node as *mut (), node as *const (), core::mem::size_of::<maple_node>() as u64) };
// Update the parent node pointer.
val = ((unsafe { (*node).__slate_anon_0.__slate_anon_0.parent }) as u64) & 255;
unsafe {
(*new_node).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr!(val | (parent as u64));
}
}

// mas_dup_alloc() - Allocate child nodes for a maple node.
// @mas: The maple state of source tree.
// @new_mas: The maple state of new tree.
// @gfp: The GFP_FLAGS to use for allocations.
//
// This function allocates child nodes for @new_mas->node during the duplication
// process. If memory allocation fails, @mas is set to -ENOMEM.
fn mas_dup_alloc(mas: *mut ma_state, new_mas: *mut ma_state, gfp: u32) {
let node: *mut maple_node = mte_to_node((unsafe { (*mas).node }) as *const maple_enode);
let new_node: *mut maple_node = mte_to_node((unsafe { (*new_mas).node }) as *const maple_enode);
let mut r#type: u32 = 0;
let mut count: u8 = 0;
let mut i: u8 = 0;
let mut new_slots: *mut *mut () = unsafe { core::mem::zeroed() };
let mut val: u64 = 0;
// Allocate memory for child nodes.
r#type = mte_node_type((unsafe { (*mas).node }) as *const maple_enode);
new_slots = ma_slots(new_node, r#type);
let __v0: u64 = (mas_data_end(mas) as i32).wrapping_add(1) as u64;
unsafe {
(*mas).node_request = __v0;
}
count = __v0 as u8;
mas_alloc_nodes(mas, gfp);
if unlikely!(mas_is_err(mas)) != 0 {
return;
}
let slots: *mut *mut () = ma_slots(node, r#type);
while (i as i32) < (count as i32) {
val = mt_slot_locked(unsafe { (*mas).tree }, slots, i) as u64;
val &= 255;
// Warning, see rcu_assign_pointer() documentation.  Since this
// is a duplication of a tree, there are no readers walking the
// tree until after the rcu_assign_pointer() call in
// mas_dup_build().
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut ()>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_850() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__15!(*new_slots.wrapping_offset((i as i32) as isize), RCU_INITIALIZER!(((mas_pop_node(mas) as u64) | val) as *mut maple_node));
if true {
break;
}
}
i = (i as i32).wrapping_add(1) as u8;
}
}

// mas_dup_build() - Build a new maple tree from a source tree
// @mas: The maple state of source tree, need to be in MAS_START state.
// @new_mas: The maple state of new tree, need to be in MAS_START state.
// @gfp: The GFP_FLAGS to use for allocations.
//
// This function builds a new tree in DFS preorder. If the memory allocation
// fails, the error code -ENOMEM will be set in @mas, and @new_mas points to the
// last node. mas_dup_free() will free the incomplete duplication of a tree.
//
// Note that the attributes of the two trees need to be exactly the same, and the
// new tree needs to be empty, otherwise -EINVAL will be set in @mas.
fn mas_dup_build(mas: *mut ma_state, new_mas: *mut ma_state, gfp: u32) {
let mut _r_a_p__v: u64 = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut r#type: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut u32 = core::ptr::addr_of_mut!(r#type) as *mut u32;
let mut root: core::mem::MaybeUninit<*mut maple_enode> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut maple_enode = core::ptr::addr_of_mut!(root) as *mut *mut maple_enode;
let mut parent: *mut maple_pnode = unsafe { core::mem::zeroed() };
let mut node: core::mem::MaybeUninit<*mut maple_node> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut maple_node = core::ptr::addr_of_mut!(node) as *mut *mut maple_node;
unsafe {
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
parent = core::ptr::null_mut::<maple_pnode>();
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if unlikely!(mt_attr((*mas).tree) != mt_attr((*new_mas).tree)) != 0 {
__v8 = true;
} else {
__v8 = unlikely!(!mtree_empty((*new_mas).tree as *const maple_tree)) != 0;
}
if __v8 {
mas_set_err(mas, 22i32.wrapping_neg() as i64);
return;
} else {
*__slate_slot_10 = mas_start(mas);
if mas_is_ptr(mas as *const ma_state) {
__v7 = true;
} else {
__v7 = mas_is_none(mas as *const ma_state);
}
'__join_0: {
if !__v7 {
*__slate_slot_12 = mt_alloc_one(gfp);
if (*__slate_slot_12).is_null() {
(*new_mas).status = 3;
mas_set_err(mas, 12i32.wrapping_neg() as i64);
return;
} else {
*__slate_slot_9 = mte_node_type((*mas).node as *const maple_enode);
*__slate_slot_10 = mt_mk_node(*__slate_slot_12 as *const maple_node, *__slate_slot_9);
(*new_mas).node = *__slate_slot_10;
(*new_mas).min = 0;
(*new_mas).max = !0;
*__slate_slot_10 = mte_mk_root(*__slate_slot_10 as *const maple_enode) as *mut maple_enode;
while true {
mas_copy_node(mas, new_mas, parent);
if !mte_is_leaf((*mas).node as *const maple_enode) {
// Only allocate child nodes for non-leaf nodes.
mas_dup_alloc(mas, new_mas, gfp);
if unlikely!(mas_is_err(mas)) != 0 {
break '__join_0;
}
} else {
// This is the last leaf node and duplication is
// completed.
if (*mas).max == !0 {
break;
} else {
// This is not the last leaf node and needs to go up.
loop {
mas_ascend(mas);
mas_ascend(new_mas);
if ((*mas).offset as i32) != (mas_data_end(mas) as i32) {
break;
}
}
// Move to the next subtree.
(*mas).offset = ((*mas).offset as i32).wrapping_add(1) as u8;
(*new_mas).offset = ((*new_mas).offset as i32).wrapping_add(1) as u8;
}
}
mas_descend(mas);
parent = ma_parent_ptr!(mte_to_node((*new_mas).node as *const maple_enode));
mas_descend(new_mas);
(*mas).offset = 0;
(*new_mas).offset = 0;
}
(*mte_to_node(*__slate_slot_10 as *const maple_enode)).__slate_anon_0.__slate_anon_0.parent = ma_parent_ptr!(mas_tree_parent!(new_mas));
// Specially handle the parent of the root node.
}
}
(*(*new_mas).tree).ma_flags = (*(*mas).tree).ma_flags;
// Make them the same height
_r_a_p__v = *__slate_slot_10 as u64;
loop {
kcsan_release!();
loop {
if true {
core::arch::asm!("", options(nostack, att_syntax));
loop {
if true {
__WRITE_ONCE!(*core::ptr::addr_of_mut!((*(*new_mas).tree).ma_root), RCU_INITIALIZER!(_r_a_p__v as *mut ()));
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
mas_empty_nodes(mas);
}
}
}

/// __mt_dup(): Duplicate an entire maple tree
/// @mt: The source maple tree
/// @new: The new maple tree
/// @gfp: The GFP_FLAGS to use for allocations
///
/// This function duplicates a maple tree in Depth-First Search (DFS) pre-order
/// traversal. It uses memcpy() to copy nodes in the source tree and allocate
/// new child nodes in non-leaf nodes. The new node is exactly the same as the
/// source node except for all the addresses stored in it. It will be faster than
/// traversing all elements in the source tree and inserting them one by one into
/// the new tree.
/// The user needs to ensure that the attributes of the source tree and the new
/// tree are the same, and the new tree needs to be an empty tree, otherwise
/// -EINVAL will be returned.
/// Note that the user needs to manually lock the source tree and the new tree.
///
/// Return: 0 on success, -ENOMEM if memory could not be allocated, -EINVAL If
/// the attributes of the two trees are different or the new tree is not an empty
/// tree.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.__mt_dup")]
extern "C" fn __mt_dup(mt: *mut maple_tree, new: *mut maple_tree, gfp: u32) -> i32 {
let mut ret: i32 = 0;
let mut mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = mt;
__t0.index = 0u64;
__t0.last = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
let mut new_mas: ma_state = {
let mut __t1: ma_state = unsafe { core::mem::zeroed() };
__t1.tree = new;
__t1.index = 0u64;
__t1.last = 0u64;
__t1.node = core::ptr::null_mut::<maple_enode>();
__t1.min = 0u64;
__t1.max = 18446744073709551615u64;
__t1.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t1.alloc = core::ptr::null_mut::<maple_node>();
__t1.node_request = 0u64;
__t1.status = 1u32;
__t1.mas_flags = 0u8;
__t1.store_type = 0u32;
__t1
};
mas_dup_build(core::ptr::addr_of_mut!(mas), core::ptr::addr_of_mut!(new_mas), gfp);
if unlikely!(mas_is_err(core::ptr::addr_of_mut!(mas))) != 0 {
ret = xa_err(mas.node as *mut ());
if ret == 12i32.wrapping_neg() {
mas_dup_free(core::ptr::addr_of_mut!(new_mas));
}
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___mt_dup_854: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, *mut maple_tree, u32) -> i32>, *mut ()>(Some(__mt_dup)) };

/// mtree_dup(): Duplicate an entire maple tree
/// @mt: The source maple tree
/// @new: The new maple tree
/// @gfp: The GFP_FLAGS to use for allocations
///
/// This function duplicates a maple tree in Depth-First Search (DFS) pre-order
/// traversal. It uses memcpy() to copy nodes in the source tree and allocate
/// new child nodes in non-leaf nodes. The new node is exactly the same as the
/// source node except for all the addresses stored in it. It will be faster than
/// traversing all elements in the source tree and inserting them one by one into
/// the new tree.
/// The user needs to ensure that the attributes of the source tree and the new
/// tree are the same, and the new tree needs to be an empty tree, otherwise
/// -EINVAL will be returned.
///
/// Return: 0 on success, -ENOMEM if memory could not be allocated, -EINVAL If
/// the attributes of the two trees are different or the new tree is not an empty
/// tree.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_dup")]
extern "C" fn mtree_dup(mt: *mut maple_tree, new: *mut maple_tree, gfp: u32) -> i32 {
let mut ret: i32 = 0;
let mut mas: ma_state = {
let mut __t0: ma_state = unsafe { core::mem::zeroed() };
__t0.tree = mt;
__t0.index = 0u64;
__t0.last = 0u64;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
};
let mut new_mas: ma_state = {
let mut __t1: ma_state = unsafe { core::mem::zeroed() };
__t1.tree = new;
__t1.index = 0u64;
__t1.last = 0u64;
__t1.node = core::ptr::null_mut::<maple_enode>();
__t1.min = 0u64;
__t1.max = 18446744073709551615u64;
__t1.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t1.alloc = core::ptr::null_mut::<maple_node>();
__t1.node_request = 0u64;
__t1.status = 1u32;
__t1.mas_flags = 0u8;
__t1.store_type = 0u32;
__t1
};
mas_lock!(core::ptr::addr_of_mut!(new_mas));
mas_lock_nested!(core::ptr::addr_of_mut!(mas), 1i32);
mas_dup_build(core::ptr::addr_of_mut!(mas), core::ptr::addr_of_mut!(new_mas), gfp);
mas_unlock!(core::ptr::addr_of_mut!(mas));
if unlikely!(mas_is_err(core::ptr::addr_of_mut!(mas))) != 0 {
ret = xa_err(mas.node as *mut ());
if ret == 12i32.wrapping_neg() {
mas_dup_free(core::ptr::addr_of_mut!(new_mas));
}
}
mas_unlock!(core::ptr::addr_of_mut!(new_mas));
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_dup_855: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, *mut maple_tree, u32) -> i32>, *mut ()>(Some(mtree_dup)) };

// __mt_destroy() - Walk and free all nodes of a locked maple tree.
// @mt: The maple tree
//
// Note: Does not handle locking.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___mt_destroy_859: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree)>, *mut ()>(Some(__mt_destroy)) };

/// mtree_destroy() - Destroy a maple tree
/// @mt: The maple tree
///
/// Frees all resources used by the tree.  Handles locking.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mtree_destroy")]
extern "C" fn mtree_destroy(mt: *mut maple_tree) {
mtree_lock!(mt);
__mt_destroy(mt);
mtree_unlock!(mt);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mtree_destroy_860: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree)>, *mut ()>(Some(mtree_destroy)) };

/// mt_find() - Search from the start up until an entry is found.
/// @mt: The maple tree
/// @index: Pointer which contains the start location of the search
/// @max: The maximum value of the search range
///
/// Takes RCU read lock internally to protect the search, which does not
/// protect the returned pointer after dropping RCU read lock.
/// See also: Documentation/core-api/maple_tree.rst
///
/// In case that an entry is found @index is updated to point to the next
/// possible entry independent whether the found entry is occupying a
/// single index or a range if indices.
///
/// Return: The entry at or after the @index or %NULL
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mt_find")]
extern "C" fn mt_find(mt: *mut maple_tree, index: *mut u64, max: u64) -> *mut () {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut entry: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(entry) as *mut *mut ();
let mut mas: core::mem::MaybeUninit<ma_state> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut ma_state = core::ptr::addr_of_mut!(mas) as *mut ma_state;
unsafe {
core::ptr::write(__slate_slot_3, {
let mut __t0: ma_state = core::mem::zeroed();
__t0.tree = mt;
__t0.index = *index;
__t0.last = *index;
__t0.node = core::ptr::null_mut::<maple_enode>();
__t0.min = 0u64;
__t0.max = 18446744073709551615u64;
__t0.sheaf = core::ptr::null_mut::<slab_sheaf>();
__t0.alloc = core::ptr::null_mut::<maple_node>();
__t0.node_request = 0u64;
__t0.status = 1u32;
__t0.mas_flags = 0u8;
__t0.store_type = 0u32;
__t0
});
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
trace_ma_read(___tp_str__6426, core::ptr::addr_of_mut!(*__slate_slot_3));
if *index > max {
return core::ptr::null_mut::<()>();
} else {
rcu_read_lock();
loop {
*__slate_slot_2 = mas_state_walk(core::ptr::addr_of_mut!(*__slate_slot_3));
if !mas_is_start(core::ptr::addr_of_mut!(*__slate_slot_3) as *const ma_state) {
break;
}
}
if unlikely!(xa_is_zero(*__slate_slot_2 as *const ())) != 0 {
*__slate_slot_2 = core::ptr::null_mut::<()>();
}
if (*__slate_slot_2).is_null() {
while mas_is_active(core::ptr::addr_of_mut!(*__slate_slot_3)) && (*__slate_slot_3).last < max {
*__slate_slot_2 = mas_next_slot(core::ptr::addr_of_mut!(*__slate_slot_3), max, false);
if !(*__slate_slot_2).is_null() {
__v0 = !xa_is_zero(*__slate_slot_2 as *const ());
} else {
__v0 = false;
}
if __v0 {
break;
}
}
if unlikely!(xa_is_zero(*__slate_slot_2 as *const ())) != 0 {
*__slate_slot_2 = core::ptr::null_mut::<()>();
}
}
rcu_read_unlock();
if likely__4!(*__slate_slot_2) != 0 {
*index = (*__slate_slot_3).last.wrapping_add(1);
}
return *__slate_slot_2;
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_str")]
static mut ___tp_str__6426: *const u8 = (b"mt_find\0".as_ptr() as *mut u8) as *const u8;

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mt_find_861: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, *mut u64, u64) -> *mut ()>, *mut ()>(Some(mt_find)) };

/// mt_find_after() - Search from the start up until an entry is found.
/// @mt: The maple tree
/// @index: Pointer which contains the start location of the search
/// @max: The maximum value to check
///
/// Same as mt_find() except that it checks @index for 0 before
/// searching. If @index == 0, the search is aborted. This covers a wrap
/// around of @index to 0 in an iterator loop.
///
/// Return: The entry at or after the @index or %NULL
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.maple_tree.mt_find_after")]
extern "C" fn mt_find_after(mt: *mut maple_tree, index: *mut u64, max: u64) -> *mut () {
if (unsafe { *index }) == 0 {
return core::ptr::null_mut::<()>();
}
mt_find(mt, index, max)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_mt_find_after_862: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut maple_tree, *mut u64, u64) -> *mut ()>, *mut ()>(Some(mt_find_after)) };

static __slate_asm_string___kmem_cache_default_args_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmem_cache_default_args_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string___mas_set_range_1: [u8; 1] = *b"\0";

static __slate_asm_string___mas_set_range_2: [u8; 56] = *b"/home/takashi/c-corpus/linux/include/linux/maple_tree.h\0";

static __slate_asm_string_cp_data_write_25: [u8; 1] = *b"\0";

static __slate_asm_string_cp_data_write_26: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_cp_is_new_root_4: [u8; 1] = *b"\0";

static __slate_asm_string_cp_is_new_root_5: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_ma_free_rcu_1: [u8; 1] = *b"\0";

static __slate_asm_string_ma_free_rcu_2: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_max_gap_0: [u8; 1] = *b"\0";

static __slate_asm_string_mas_max_gap_1: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_new_root_17: [u8; 1] = *b"\0";

static __slate_asm_string_mas_new_root_18: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_next_slot_13: [u8; 1] = *b"\0";

static __slate_asm_string_mas_next_slot_14: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_nomem_2: [u8; 1] = *b"\0";

static __slate_asm_string_mas_nomem_3: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_parent_gap_6: [u8; 1] = *b"\0";

static __slate_asm_string_mas_parent_gap_7: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_parent_type_1: [u8; 1] = *b"\0";

static __slate_asm_string_mas_parent_type_2: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_pop_node_10: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_pop_node_9: [u8; 1] = *b"\0";

static __slate_asm_string_mas_prealloc_calc_2: [u8; 1] = *b"\0";

static __slate_asm_string_mas_prealloc_calc_3: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_prev_slot_13: [u8; 1] = *b"\0";

static __slate_asm_string_mas_prev_slot_14: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_set_parent_slots_0: [u8; 1] = *b"\0";

static __slate_asm_string_mas_set_parent_slots_1: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_store_2: [u8; 1] = *b"\0";

static __slate_asm_string_mas_store_3: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_store_prealloc_2: [u8; 1] = *b"\0";

static __slate_asm_string_mas_store_prealloc_3: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_topiary_replace_25: [u8; 1] = *b"\0";

static __slate_asm_string_mas_topiary_replace_26: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_topiary_replace_7: [u8; 1] = *b"\0";

static __slate_asm_string_mas_topiary_replace_8: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_wr_slot_store_23: [u8; 1] = *b"\0";

static __slate_asm_string_mas_wr_slot_store_24: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mas_wr_store_entry_8: [u8; 1] = *b"\0";

static __slate_asm_string_mas_wr_store_entry_9: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mt_set_height_2: [u8; 1] = *b"\0";

static __slate_asm_string_mt_set_height_3: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mte_set_pivot_0: [u8; 1] = *b"\0";

static __slate_asm_string_mte_set_pivot_1: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mtree_alloc_cyclic_1: [u8; 1] = *b"\0";

static __slate_asm_string_mtree_alloc_cyclic_2: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mtree_alloc_range_5: [u8; 1] = *b"\0";

static __slate_asm_string_mtree_alloc_range_6: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mtree_alloc_rrange_5: [u8; 1] = *b"\0";

static __slate_asm_string_mtree_alloc_rrange_6: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mtree_insert_range_5: [u8; 1] = *b"\0";

static __slate_asm_string_mtree_insert_range_6: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_mtree_store_range_2: [u8; 1] = *b"\0";

static __slate_asm_string_mtree_store_range_3: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

static __slate_asm_string_spanning_sib_14: [u8; 1] = *b"\0";

static __slate_asm_string_spanning_sib_15: [u8; 46] = *b"/home/takashi/c-corpus/linux/lib/maple_tree.c\0";

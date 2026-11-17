
//! SPDX-License-Identifier: GPL-2.0-only
//! SPDX-License-Identifier: GPL-2.0
//! SPDX-License-Identifier: GPL-2.0
//! These attributes define new context lock (Clang: capability) types.
//! Internal only.
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-aligned-function-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html#index-aligned-type-attribute
//! gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-aligned-variable-attribute
//! Turn individual warnings and errors on and off locally, depending
//! on version.
//! SPDX-License-Identifier: GPL-2.0-or-later
//!
//! libata-sff.c - helper library for PCI IDE BMDMA
//!
//! Copyright 2003-2006 Red Hat, Inc.  All rights reserved.
//! Copyright 2003-2006 Jeff Garzik
//!
//! libata documentation is available via 'make {ps|pdf}docs',
//! as Documentation/driver-api/libata.rst
//!
//! Hardware documentation available from http://www.t13.org/ and
//! http://www.sata-io.org/
//! @a is a power of 2 value
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
//! ARRAY_SIZE - get the number of elements in array @arr
//! @arr: array to be sized
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
//! do_div - returns 2 values: calculate remainder and update new dividend
//! @n: uint64_t dividend (will be updated)
//! @base: uint32_t divisor
//!
//! Summary:
//! ``uint32_t remainder = n % base;``
//! ``n = n / base;``
//!
//! Return: (uint32_t)remainder
//!
//! NOTE: macro parameter @n is evaluated multiple times,
//! beware of side effects!
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
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
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
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! used to decode ioctl numbers..
//! The inline keyword gives the compiler room to decide to inline, or
//! not inline a function as it sees best.  However, as these functions
//! are called in both __init and non-__init functions, if they are not
//! inlined we will end up with a section mismatch error (of the type of
//! freeable items not being freed).  So we must use __always_inline here
//! to fix the problem.  If other functions in the future also end up in
//! this situation they will also need to be annotated as __always_inline
//! Wrap the arch provided IRQ routines to provide appropriate checks.
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
//! Used in tsk->__state:
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
//! Keep unconverted code working
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Generate SD flag indexes
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
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! no-op
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
//! These flags used only by the kernel as part of the
//! irq handling routines.
//!
//! IRQF_SHARED - allow sharing the irq among several devices
//! IRQF_PROBE_SHARED - set by callers when they expect sharing mismatches to occur
//! IRQF_TIMER - Flag to mark this interrupt as timer interrupt
//! IRQF_PERCPU - Interrupt is per cpu
//! IRQF_NOBALANCING - Flag to exclude this interrupt from irq balancing
//! IRQF_IRQPOLL - Interrupt is used for polling (only the interrupt that is
//!                registered first in a shared interrupt is considered for
//!                performance reasons)
//! IRQF_ONESHOT - Interrupt is not reenabled after the hardirq handler finished.
//!                Used by threaded interrupts which need to keep the
//!                irq line disabled until the threaded handler has been run.
//! IRQF_NO_SUSPEND - Do not disable this IRQ during suspend.  Does not guarantee
//!                   that this interrupt will wake the system from a suspended
//!                   state.  See Documentation/power/suspend-and-interrupts.rst
//! IRQF_FORCE_RESUME - Force enable it on resume even if IRQF_NO_SUSPEND is set
//! IRQF_NO_THREAD - Interrupt cannot be threaded
//! IRQF_EARLY_RESUME - Resume IRQ early during syscore instead of at device
//!                resume time.
//! IRQF_COND_SUSPEND - If the IRQ is shared with a NO_SUSPEND user, execute this
//!                interrupt handler after suspending interrupts. For system
//!                wakeup devices users need to implement wakeup detection in
//!                their interrupt handlers.
//! IRQF_NO_AUTOEN - Don't enable IRQ or NMI automatically when users request it.
//!                Users will enable it explicitly by enable_irq() or enable_nmi()
//!                later.
//! IRQF_NO_DEBUG - Exclude from runnaway detection for IPI and similar handlers,
//! 		   depends on IRQF_PERCPU.
//! IRQF_COND_ONESHOT - Agree to do IRQF_ONESHOT if already set for a shared
//!                 interrupt.
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! Must be a compile-time constant, so implement it as a macro
//! Bits set in the VMA until the stack is in its final location
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! iterate over multi-page bvec
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! The Well Known LUNS (SAM-3) in our int representation of a LUN
//! This macro is used to tag functions as "printf-like" because
//! some compilers (like GCC) can catch printf format string problems.
//! Value returned by acpi_os_get_thread_id. There is no standard "thread_id"
//! across operating systems or even the various UNIX systems. Since ACPICA
//! only needs the thread ID as a unique thread identifier, we use a u64
//! as the only common data type - it will accommodate any type of pointer or
//! any type of integer. It is up to the host-dependent OSL to cast the
//! native thread ID type to a u64 (in acpi_os_get_thread_id).
//! For values, see Producer/Consumer above
//! The maximum expected number of CSI-2 data lanes.
//!
//! This number is not expected to ever have to be equal to or greater than the
//! number of bits in an unsigned long variable, but if it needs to be increased
//! above that limit, code will need to be adjusted accordingly.
//! DWORD 3
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
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
// The below are used to annotate code being checked. Internal only.
// The "assert_capability" attribute is a bit confusingly named. It does not
// generate a check. Instead, it tells the analysis to *assume* the capability
// is held. This is used for augmenting runtime assertions, that can then help
// with patterns beyond the compiler's static reasoning abilities.
// __guarded_by() - struct member and globals attribute, declares variable
//                  only accessible within active context
// @...: context lock instance pointer(s)
//
// Declares that the struct member or global variable is only accessible within
// the context entered by the given context lock(s). Read operations on the data
// require shared access to at least one of the context locks, while write
// operations require exclusive access to all listed context locks.
//
// .. code-block:: c
//
// 	struct some_state {
// 		spinlock_t lock;
// 		long counter __guarded_by(&lock);
// 	};
//
// 	struct some_state {
// 		spinlock_t lock1, lock2;
// 		long counter __guarded_by(&lock1, &lock2);
// 	};
// context_lock_struct() - declare or define a context lock struct
// @name: struct name
//
// Helper to declare or define a struct type that is also a context lock.
//
// .. code-block:: c
//
// 	context_lock_struct(my_handle) {
// 		int foo;
// 		long bar;
// 	};
//
// 	struct some_state {
// 		...
// 	};
// 	// ... declared elsewhere ...
// 	context_lock_struct(some_state);
//
// Note: The implementation defines several helper functions that can acquire
// and release the context lock.
// enable_context_analysis() - re-enables context analysis
//
// Re-enables context analysis. Must be paired with a prior
// disable_context_analysis().
// __no_context_analysis - function attribute, disables context analysis
//
// Function attribute denoting that context analysis is disabled for the
// whole function. Prefer use of `context_unsafe()` where possible.
// context_unsafe() - disable context checking for contained code
//
// Disables context checking for contained statements or expression.
//
// .. code-block:: c
//
// 	struct some_data {
// 		spinlock_t lock;
// 		int counter __guarded_by(&lock);
// 	};
//
// 	int foo(struct some_data *d)
// 	{
// 		// ...
// 		// other code that is still checked ...
// 		// ...
// 		return context_unsafe(d->counter);
// 	}
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
// Severity used in pragma directives
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
// modpost check for section mismatches during the kernel build.
// A section mismatch happens when there are references from a
// code or data section to an init section (both code or data).
// The init sections are (for most archs) discarded by the kernel
// when early init has completed so all such references are potential bugs.
// For exit sections the same issue exists.
//
// The following markers are used for the cases where the reference to
// the *init / *exit section (code or data) is valid and will teach
// modpost not to issue a warning.  Intended semantics is that a code or
// data tagged __ref* can reference code or data from init section without
// producing a warning (of course, no warning does not mean code is
// correct, so optimally document why the __ref is needed and why it's OK).
//
// The markers follow same syntax rules as __init / __initdata.
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
// "acpi" ACPI via MSR
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
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
// Non-serializing WRMSR
// INVLPGB and TLBSYNC instructions supported
// "wbnoinvd" WBNOINVD instruction
// "hwp" Intel Hardware P-states
// "ospke" OS Protection Keys Enable
// Shadow stack
// "la57" 5-level page tables
// "rdpid" RDPID instruction
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
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
// max_t - return maximum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Not a typewriter
// Text file busy
// No space left on device
// Math result not representable
// Invalid system call number
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
// No static inline type checking - see Subtlety (1) above.
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
// This function is needed by all SMP systems. It must _always_ be valid
// from the initial startup.
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
// Don't call further
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
// Need to take variadic arguments even though we don't use them, as dev_fmt()
// may only just have been expanded and may result in multiple arguments.
// Dummy dev_printk for disabled debugging statements to use whilst maintaining
// gcc's format checking.
// Generate SD flag bits
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
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// See the comment in asm-generic/io.h about ioremap_np().
// Reg. Level Programming Interface
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// DWORD 5
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
macro_rules! ATA_PRIMARY_IRQ {
() => {
14i32
};
}

macro_rules! ATA_SECONDARY_IRQ {
() => {
15i32
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

macro_rules! IRQ_RETVAL {
($x:expr) => {
if $x { 1i32 } else { 0i32 }
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

macro_rules! PFN_PHYS {
($x:expr) => {
(($x) as u64) << 12i32
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

/// __acquire() - function to acquire context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x exclusively.
macro_rules! __acquire__2 {
($x:expr) => {
__acquire_ctx_lock__2(($x) as *const spinlock);
};
}

/// __acquire_shared() - function to acquire context lock shared
/// @x: context lock instance pointer
///
/// No-op function that acquires the given context lock instance @x with shared
/// access.
macro_rules! __acquire_shared__2 {
($x:expr) => {
__acquire_shared_ctx_lock__8(($x) as *const srcu_struct);
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
macro_rules! __native_word__7 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
() => {
((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release__2 {
($x:expr) => {
__release_ctx_lock__1(($x) as *const raw_spinlock);
};
}

/// __release_shared() - function to release context lock shared
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x with shared
/// access.
macro_rules! __release_shared__2 {
($x:expr) => {
__release_shared_ctx_lock__8(($x) as *const srcu_struct);
};
}

/// Are two types/vars the same type (ignoring qualifiers)?
macro_rules! __same_type {
() => {
0i32
};
}

macro_rules! __timer_init {
($_timer:expr, $_fn:expr, $_flags:expr) => {
unsafe { timer_init_key($_timer, $_fn, ($_flags) as u32, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>()) };
};
}

macro_rules! __va {
($x:expr) => {
(($x) as u64).wrapping_add(unsafe { page_offset_base }) as *mut ()
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
macro_rules! __verify_pcpu_ptr__3 {
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
macro_rules! data_race__2 {
($expr:expr) => {
{
let mut __v: u32;
__kcsan_disable_current();
disable_context_analysis!();
let __v2: u32 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
}
};
}

/// disable_context_analysis() - disables context analysis
///
/// Disables context analysis. Must be paired with a later
/// enable_context_analysis().
macro_rules! disable_context_analysis {
() => {
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

macro_rules! likely__6 {
($x:expr) => {
!$x.is_null() as i64
};
}

macro_rules! lockdep_assert_held__2 {
($l:expr) => {
__assume_ctx_lock__2(($l) as *const spinlock);
};
}

macro_rules! ndelay {
($x:expr) => {
ndelay(($x) as u64);
};
}

macro_rules! offsetof {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, dev) as u64
};
}

macro_rules! offsetof__90 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, sff_pio_task.work) as u64
};
}

macro_rules! page_address {
($page:expr) => {
lowmem_page_address($page)
};
}

macro_rules! page_to_virt {
($x:expr) => {
__va!(PFN_PHYS!(((unsafe { $x.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64))
};
}

macro_rules! pci_resource_end {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).end }
};
}

macro_rules! pci_resource_len {
($dev:expr, $bar:expr) => {
{
let __v: u64 = if pci_resource_end!($dev, $bar) != 0 {
resource_size(pci_resource_n!($dev, $bar) as *const resource)
} else {
0
};
__v
}
};
}

/// These helpers provide future and backwards compatibility
/// for accessing popular PCI BAR info
macro_rules! pci_resource_n {
($dev:expr, $bar:expr) => {
unsafe { (*$dev).resource.as_mut_ptr() as *mut resource }.wrapping_offset(($bar) as isize)
};
}

macro_rules! pci_resource_start {
($dev:expr, $bar:expr) => {
unsafe { (*pci_resource_n!($dev, $bar)).start }
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

macro_rules! raw_spin_lock_irq {
($lock:expr) => {
unsafe { _raw_spin_lock_irq($lock) };
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

macro_rules! raw_spin_unlock_irq {
($lock:expr) => {
unsafe { _raw_spin_unlock_irq($lock) };
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

macro_rules! spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(spinlock_check($lock), $flags);
__release__2!(spinlock_check($lock));
__acquire__2!($lock);
if true {
break;
}
}
};
}

macro_rules! static_branch_unlikely__8 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const9(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__9 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const10(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
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

macro_rules! static_branch_unlikely__10 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const12(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__11 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const13(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! static_branch_unlikely__12 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const8(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
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

macro_rules! static_branch_unlikely__6 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const6(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__5 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const5(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__7 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const7(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! static_branch_unlikely__13 {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = arch_static_branch__const11(unsafe { core::ptr::addr_of_mut!((*$x).key) }, false);
__v = __v2 as i64;
__v
}
};
}

macro_rules! test_bit__2 {
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

macro_rules! to_pci_dev {
($n:expr) => {
{
let mut __v: *mut pci_dev;
__v = ((unsafe { ((($n) as *mut ()) as *const u8).wrapping_offset(-(offsetof!(pci_dev) as isize)) }) as *mut ()) as *mut pci_dev;
__v
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

macro_rules! unlikely__3 {
($x:expr) => {
(($x) != 0) as i64
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
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut this_cpu_off: u64;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut __cpu_online_mask: cpumask;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut cpuinfo_op: seq_operations;
    static mut cpu_number: i32;
    static mut runqueues: rq;
    static mut sighand_cachep: *mut kmem_cache;
    static mut ata_dummy_port_ops: ata_port_operations;
    static mut sata_deb_timing_normal: [u32; 0];
    static mut sata_deb_timing_hotplug: [u32; 0];
    static mut ata_base_port_ops: ata_port_operations;
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_ata_tf_load: static_call_key;
    static mut __tracepoint_ata_tf_load: tracepoint;
    static mut __SCK__tp_func_ata_exec_command: static_call_key;
    static mut __tracepoint_ata_exec_command: tracepoint;
    static mut __SCK__tp_func_ata_bmdma_setup: static_call_key;
    static mut __tracepoint_ata_bmdma_setup: tracepoint;
    static mut __SCK__tp_func_ata_bmdma_start: static_call_key;
    static mut __tracepoint_ata_bmdma_start: tracepoint;
    static mut __SCK__tp_func_ata_bmdma_stop: static_call_key;
    static mut __tracepoint_ata_bmdma_stop: tracepoint;
    static mut __SCK__tp_func_ata_bmdma_status: static_call_key;
    static mut __tracepoint_ata_bmdma_status: tracepoint;
    static mut __SCK__tp_func_ata_sff_hsm_state: static_call_key;
    static mut __tracepoint_ata_sff_hsm_state: tracepoint;
    static mut __SCK__tp_func_ata_sff_hsm_command_complete: static_call_key;
    static mut __tracepoint_ata_sff_hsm_command_complete: tracepoint;
    static mut __SCK__tp_func_ata_sff_port_intr: static_call_key;
    static mut __tracepoint_ata_sff_port_intr: tracepoint;
    static mut __SCK__tp_func_ata_sff_pio_transfer_data: static_call_key;
    static mut __tracepoint_ata_sff_pio_transfer_data: tracepoint;
    static mut __SCK__tp_func_atapi_pio_transfer_data: static_call_key;
    static mut __tracepoint_atapi_pio_transfer_data: tracepoint;
    static mut __SCK__tp_func_atapi_send_cdb: static_call_key;
    static mut __tracepoint_atapi_send_cdb: tracepoint;
    static mut __SCK__tp_func_ata_sff_flush_pio_task: static_call_key;
    static mut __tracepoint_ata_sff_flush_pio_task: tracepoint;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn ____wrong_branch_error() -> bool;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_57();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn ___migrate_enable();
    fn rcu_is_watching() -> bool;
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn alloc_workqueue_noprof(arg0: *const u8, arg1: u32, arg2: i32, ...) -> *mut workqueue_struct;
    fn destroy_workqueue(arg0: *mut workqueue_struct);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn queue_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn cancel_delayed_work_sync(arg0: *mut delayed_work) -> bool;
    fn __compiletime_assert_167();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn _dev_err(arg0: *const device, arg1: *const u8, ...);
    fn _dev_warn(arg0: *const device, arg1: *const u8, ...);
    fn devres_open_group(arg0: *mut device, arg1: *mut (), arg2: u32) -> *mut ();
    fn devres_remove_group(arg0: *mut device, arg1: *mut ());
    fn devres_release_group(arg0: *mut device, arg1: *mut ()) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_388();
    fn __compiletime_assert_409();
    fn dev_driver_string(arg0: *const device) -> *const u8;
    fn devm_request_threaded_irq(arg0: *mut device, arg1: u32, arg2: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, arg3: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, arg4: u64, arg5: *const u8, arg6: *mut ()) -> i32;
    fn ioread8(arg0: *const ()) -> u32;
    fn ioread16(arg0: *const ()) -> u32;
    fn iowrite8(arg0: u8, arg1: *mut ());
    fn iowrite32(arg0: u32, arg1: *mut ());
    fn ioread16_rep(arg0: *const (), arg1: *mut (), arg2: u64);
    fn ioread32_rep(arg0: *const (), arg1: *mut (), arg2: u64);
    fn iowrite16_rep(arg0: *mut (), arg1: *const (), arg2: u64);
    fn iowrite32_rep(arg0: *mut (), arg1: *const (), arg2: u64);
    fn pci_read_config_byte(arg0: *const pci_dev, arg1: i32, arg2: *mut u8) -> i32;
    fn pcim_enable_device(arg0: *mut pci_dev) -> i32;
    fn pcim_pin_device(arg0: *mut pci_dev);
    fn pci_set_master(arg0: *mut pci_dev);
    fn pcim_iomap_table(arg0: *mut pci_dev) -> *const *mut ();
    fn pcim_iomap_regions(arg0: *mut pci_dev, arg1: i32, arg2: *const u8) -> i32;
    fn dmam_alloc_attrs(arg0: *mut device, arg1: u64, arg2: *mut u64, arg3: u32, arg4: u64) -> *mut ();
    fn dma_set_mask(arg0: *mut device, arg1: u64) -> i32;
    fn dma_set_coherent_mask(arg0: *mut device, arg1: u64) -> i32;
    fn __bad_udelay();
    fn __bad_ndelay();
    fn __udelay(arg0: u64);
    fn __ndelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn cdrom_multisession(arg0: *mut cdrom_device_info, arg1: *mut cdrom_multisession) -> i32;
    fn ata_std_prereset(arg0: *mut ata_link, arg1: u64) -> i32;
    fn ata_std_postreset(arg0: *mut ata_link, arg1: *mut u32);
    fn ata_host_alloc_pinfo(arg0: *mut device, arg1: *const *const ata_port_info, arg2: i32) -> *mut ata_host;
    fn ata_host_start(arg0: *mut ata_host) -> i32;
    fn ata_host_register(arg0: *mut ata_host, arg1: *const scsi_host_template) -> i32;
    fn ata_link_offline(arg0: *mut ata_link) -> bool;
    fn ata_msleep(arg0: *mut ata_port, arg1: u32);
    fn ata_port_classify(arg0: *mut ata_port, arg1: *const ata_taskfile) -> u32;
    fn ata_qc_complete(arg0: *mut ata_queued_cmd);
    fn sata_scr_valid(arg0: *mut ata_link) -> i32;
    fn sata_link_hardreset(arg0: *mut ata_link, arg1: *const u32, arg2: u64, arg3: *mut bool, arg4: Option<unsafe extern "C" fn(*mut ata_link) -> i32>) -> i32;
    fn ata_acpi_gtm(arg0: *mut ata_port, arg1: *mut ata_acpi_gtm) -> i32;
    fn ata_port_freeze(arg0: *mut ata_port) -> i32;
    fn ata_eh_thaw_port(arg0: *mut ata_port);
    fn ata_std_error_handler(arg0: *mut ata_port);
    fn ata_ehi_push_desc(arg0: *mut ata_eh_info, arg1: *const u8, ...);
    fn ata_port_desc(arg0: *mut ata_port, arg1: *const u8, ...);
    fn __traceiter_ata_tf_load(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile) -> i32;
    fn __SCT__tp_func_ata_tf_load(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile) -> i32;
    fn __compiletime_assert_654();
    fn __traceiter_ata_exec_command(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile, arg3: u32) -> i32;
    fn __SCT__tp_func_ata_exec_command(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile, arg3: u32) -> i32;
    fn __compiletime_assert_661();
    fn __traceiter_ata_bmdma_setup(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile, arg3: u32) -> i32;
    fn __SCT__tp_func_ata_bmdma_setup(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile, arg3: u32) -> i32;
    fn __compiletime_assert_668();
    fn __traceiter_ata_bmdma_start(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile, arg3: u32) -> i32;
    fn __SCT__tp_func_ata_bmdma_start(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile, arg3: u32) -> i32;
    fn __compiletime_assert_675();
    fn __traceiter_ata_bmdma_stop(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile, arg3: u32) -> i32;
    fn __SCT__tp_func_ata_bmdma_stop(arg0: *mut (), arg1: *mut ata_port, arg2: *const ata_taskfile, arg3: u32) -> i32;
    fn __compiletime_assert_682();
    fn __traceiter_ata_bmdma_status(arg0: *mut (), arg1: *mut ata_port, arg2: u32) -> i32;
    fn __SCT__tp_func_ata_bmdma_status(arg0: *mut (), arg1: *mut ata_port, arg2: u32) -> i32;
    fn __compiletime_assert_689();
    fn __traceiter_ata_sff_hsm_state(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u8) -> i32;
    fn __SCT__tp_func_ata_sff_hsm_state(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u8) -> i32;
    fn __compiletime_assert_801();
    fn __traceiter_ata_sff_hsm_command_complete(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u8) -> i32;
    fn __SCT__tp_func_ata_sff_hsm_command_complete(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u8) -> i32;
    fn __compiletime_assert_808();
    fn __traceiter_ata_sff_port_intr(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u8) -> i32;
    fn __SCT__tp_func_ata_sff_port_intr(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u8) -> i32;
    fn __compiletime_assert_815();
    fn __traceiter_ata_sff_pio_transfer_data(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u32, arg3: u32) -> i32;
    fn __SCT__tp_func_ata_sff_pio_transfer_data(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u32, arg3: u32) -> i32;
    fn __compiletime_assert_822();
    fn __traceiter_atapi_pio_transfer_data(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u32, arg3: u32) -> i32;
    fn __SCT__tp_func_atapi_pio_transfer_data(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u32, arg3: u32) -> i32;
    fn __compiletime_assert_829();
    fn __traceiter_atapi_send_cdb(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u32, arg3: u32) -> i32;
    fn __SCT__tp_func_atapi_send_cdb(arg0: *mut (), arg1: *mut ata_queued_cmd, arg2: u32, arg3: u32) -> i32;
    fn __compiletime_assert_836();
    fn __traceiter_ata_sff_flush_pio_task(arg0: *mut (), arg1: *mut ata_port) -> i32;
    fn __SCT__tp_func_ata_sff_flush_pio_task(arg0: *mut (), arg1: *mut ata_port) -> i32;
    fn __compiletime_assert_843();
    fn ata_wait_ready(arg0: *mut ata_link, arg1: u64, arg2: Option<unsafe extern "C" fn(*mut ata_link) -> i32>) -> i32;
    fn __builtin_unreachable();
    fn __compiletime_assert_863();
    fn __compiletime_assert_866();
    fn __compiletime_assert_873();
    fn __compiletime_assert_879();
    fn __compiletime_assert_882();
    fn __compiletime_assert_883();
    fn __compiletime_assert_886();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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


#[inline(always)]
unsafe extern "C" fn __WARN_validate_printf(fmt: *const u8, __va_args: ...) {
}

// inhibit tail-call optimization
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


#[inline(always)]
fn __acquire_ctx_lock__1(var: &raw_spinlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__1(var: &raw_spinlock) {
}

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[inline(always)]
fn __release_ctx_lock__1(var: *const raw_spinlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__1(var: &raw_spinlock) {
}

#[inline(always)]
fn __assume_ctx_lock__1(var: &raw_spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
if unlikely!(preempt_count_dec_and_test!()) != 0 {
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
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_6: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

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
struct io_bitmap {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_tf_load, const 0 as u8, label {
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
fn arch_static_branch__const2(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_exec_command, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_bmdma_setup, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_bmdma_start, const 0 as u8, label {
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
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn arch_static_branch__const5(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_bmdma_stop, const 0 as u8, label {
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
fn arch_static_branch__const6(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_bmdma_status, const 0 as u8, label {
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
fn arch_static_branch__const7(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_sff_hsm_state, const 0 as u8, label {
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
struct pollfd {
}


#[inline(always)]
fn arch_static_branch__const8(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_sff_hsm_command_complete, const 0 as u8, label {
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
fn arch_static_branch__const9(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_sff_port_intr, const 0 as u8, label {
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
fn arch_static_branch__const10(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_sff_pio_transfer_data, const 0 as u8, label {
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn arch_static_branch__const11(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_atapi_pio_transfer_data, const 0 as u8, label {
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
fn arch_static_branch__const12(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_atapi_send_cdb, const 0 as u8, label {
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
fn arch_static_branch__const13(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_ata_sff_flush_pio_task, const 0 as u8, label {
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
    __slate_bits_0: __slate_bits::__SlateBits104U0,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits105U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_64_types_h_21 {
    pte: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
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
#[inline(always)]
fn cpu_max_bits_warn(cpu: u32, bits: u32) {
}

#[inline(always)]
fn cpumask_check(cpu: u32) -> u32 {
cpu_max_bits_warn(cpu, 64u32);
cpu
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit__2!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
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

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
    __slate_bits_0: __slate_bits::__SlateBits124U0,
    pkru: u32,
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
}


#[inline(always)]
fn __acquire_ctx_lock__2(var: *const spinlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__2(var: &spinlock) {
}

#[inline(always)]
fn __release_ctx_lock__2(var: &spinlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__2(var: &spinlock) {
}

#[inline(always)]
fn __assume_ctx_lock__2(var: *const spinlock) {
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
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
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
fn __acquire_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __release_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __assume_ctx_lock__3(var: &rwlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
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
    __slate_bits_0: __slate_bits::__SlateBits161U0,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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


#[inline(always)]
fn spin_unlock_irqrestore(lock: *mut spinlock, flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }, flags);
}

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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[inline(always)]
fn __acquire_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __release_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __release_shared_ctx_lock__4(var: &mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[inline(always)]
fn __assume_ctx_lock__4(var: &mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
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
struct seqcount {
    sequence: u32,
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
struct seqcount_spinlock {
    seqcount: seqcount,
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


#[inline(always)]
fn __acquire_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __release_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__5(var: &seqlock) {
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


#[inline(always)]
fn __assume_ctx_lock__5(var: &seqlock) {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
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
struct uprobe {
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
struct uprobe_xol_ops {
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
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_9 {
    val: u32,
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
union unwind_task_id {
    __slate_anon_0: __SlateRecord_unwind_deferred_types_h_28,
    id: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits290U0,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
// plug has ts that needs updating
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
let mut __UNIQUE_ID_guard_129: __SlateRecord_preempt_h_496 = class_preempt_constructor();
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_129) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || class_preempt_destructor(__slate_cleanup_ptr_3 as *mut _));
if unlikely!((unsafe { (*p).cpus_ptr }) != ((unsafe { core::ptr::addr_of_mut!((*p).cpus_mask) }) as *const cpumask)) != 0 {
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
struct user_namespace {
}


fn __migrate_disable() {
let p: *mut task_struct = get_current();
if (unsafe { (*p).migration_disabled }) != 0 {
unsafe {
(*p).migration_disabled = ((*p).migration_disabled as i32).wrapping_add(1) as u16;
}
return;
}
let mut __UNIQUE_ID_guard_130: __SlateRecord_preempt_h_496 = class_preempt_constructor();
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_130) as *mut _;
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

#[repr(C)]
#[derive(Clone, Copy)]
struct ns_common {
}


#[inline(always)]
fn migrate_enable() {
__migrate_enable();
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


#[inline(always)]
fn __acquire_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[inline(always)]
fn __release_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[inline(always)]
fn __release_shared_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[inline(always)]
fn __assume_ctx_lock__6(var: &__ctx_lock_RCU) {
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
struct __ctx_lock_RCU {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


// Defined as a macro as it is a very low level header included from
// areas that don't even know about current.  This gives the rcu_read_lock()
// nesting depth, but makes sense only if CONFIG_PREEMPT_RCU -- in other
// types of kernel builds, the rcu_read_lock() nesting depth is unknowable.
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


#[inline(always)]
fn __acquire_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __release_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __release_shared_ctx_lock__7(var: &rw_semaphore) {
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


#[inline(always)]
fn __assume_ctx_lock__7(var: &rw_semaphore) {
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
struct wake_irq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
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
struct rcu_segcblist {
    head: *mut callback_head,
    tails: [*mut *mut callback_head; 4],
    gp_seq: [rcu_gp_seq; 4],
    len: i64,
    seglen: [i64; 4],
    flags: u8,
}


#[inline(always)]
fn __acquire_ctx_lock__8(var: &srcu_struct) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__8(var: *const srcu_struct) {
}

#[inline(always)]
fn __release_ctx_lock__8(var: &srcu_struct) {
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


#[inline(always)]
fn __release_shared_ctx_lock__8(var: *const srcu_struct) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
}


#[inline(always)]
fn __assume_ctx_lock__8(var: &srcu_struct) {
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


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__7!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
} else {
let mut __v13: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__3!();
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
__acquire_shared__2!(ssp);
scp
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared__2!(ssp);
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
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
let mut __v11: *mut __SlateRecord_types_h_197;
__verify_pcpu_ptr__3!();
let mut __v13: *mut __SlateRecord_types_h_197;
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
let __v16: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!((*scp).srcu_unlocks) }) as u64);
tcp_ptr__ = __v16;
__v13 = tcp_ptr__ as *mut __SlateRecord_types_h_197;
__v11 = __v13;
atomic_long_inc(__v11);
}
// Z, and implicit RCU reader.
// Z, and implicit RCU reader.
}

fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


// Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
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

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


fn srcu_read_unlock_fast_notrace(ssp: *mut srcu_struct, scp: *mut srcu_ctr) {
srcu_check_read_flavor(ssp, 4i32);
__srcu_read_unlock_fast(ssp, scp);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_srcu_h_645 {
    lock: *mut srcu_struct,
    scp: *mut srcu_ctr,
}


#[inline(always)]
fn class_srcu_fast_notrace_destructor(_T: *mut __SlateRecord_srcu_h_645) {
srcu_read_unlock_fast_notrace(unsafe { (*_T).lock }, unsafe { (*_T).scp });
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
struct arch_uprobe {
    __slate_anon_0: __SlateRecord_uprobes_h_31,
    ops: *const uprobe_xol_ops,
    __slate_anon_2: __SlateRecord_uprobes_h_38,
    flags: u64,
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
struct arch_uprobe_task {
    saved_scratch_register: u64,
    saved_trap_nr: u32,
    saved_tf: u32,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits416U0,
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
#[inline(always)]
fn PageSlab(page: *const page) -> i32 {
(data_race__2!((unsafe { (*page).__slate_anon_2.page_type }) >> 24i32) == 245) as i32
}

#[inline(always)]
fn __acquire_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
}


#[inline(always)]
fn __release_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__9(var: &local_lock) {
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


#[inline(always)]
fn __assume_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_lock {
}


#[inline(always)]
fn __acquire_ctx_lock__10(var: &local_trylock) {
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


#[inline(always)]
fn __acquire_shared_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __release_ctx_lock__10(var: &local_trylock) {
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


#[inline(always)]
fn __release_shared_ctx_lock__10(var: &local_trylock) {
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


#[inline(always)]
fn __assume_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_device_id {
    vendor: u32,
    device: u32,
    subvendor: u32,
    subdevice: u32,
    class: u32,
    class_mask: u32,
    driver_data: u64,
    override_only: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iocontext_h_83 {
    q_node: list_head,
    __rcu_icq_cache: *mut kmem_cache,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_iocontext_h_87 {
    ioc_node: hlist_node,
    __rcu_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct resource {
    start: u64,
    end: u64,
    name: *const u8,
    flags: u64,
    desc: u64,
    parent: *mut resource,
    sibling: *mut resource,
    child: *mut resource,
}


// Resource type
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
fn resource_size(res: *const resource) -> u64 {
unsafe { (*res).end }.wrapping_sub(unsafe { (*res).start }).wrapping_add(1)
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
struct fiemap_extent_info {
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
    __slate_bits_0: __slate_bits::__SlateBits470U0,
    vm_ops: *const vm_operations_struct,
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
struct posix_acl {
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
struct io_uring_cmd {
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
    __slate_bits_0: __slate_bits::__SlateBits472U0,
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
    __slate_bits_0: __slate_bits::__SlateBits497U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits497U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits497U2,
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
    __slate_bits_3: __slate_bits::__SlateBits497U3,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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


#[inline(always)]
fn __acquire_ctx_lock__11(var: &__context_bitlock) {
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
struct proc_dir_entry {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __assume_ctx_lock__11(var: &__context_bitlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __context_bitlock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hotplug_slot {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hlist_bl_node {
    next: *mut hlist_bl_node,
    pprev: *mut *mut hlist_bl_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_bwctrl_data {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pcie_link_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_sriov {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rcec_ea {
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
union __SlateRecord_pci_h_547 {
    sriov: *mut pci_sriov,
    physfn: *mut pci_dev,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gpio_desc {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
}


#[inline(always)]
fn pagefault_disabled_inc() {
let __v0: *mut task_struct = get_current();
let __v1: i32 = unsafe { (*__v0).pagefault_disabled };
unsafe {
(*__v0).pagefault_disabled = __v1.wrapping_add(1);
}
}

#[inline(always)]
fn pagefault_disabled_dec() {
let __v0: *mut task_struct = get_current();
let __v1: i32 = unsafe { (*__v0).pagefault_disabled };
unsafe {
(*__v0).pagefault_disabled = __v1.wrapping_sub(1);
}
}

fn pagefault_disable() {
pagefault_disabled_inc();
// make sure to have issued the store before a pagefault
// can hit.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
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


fn pagefault_enable() {
// make sure to issue those last loads/stores before enabling
// the pagefault handler again.
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
pagefault_disabled_dec();
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
struct disk_stats {
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
union __SlateRecord_blk_types_h_239 {
    bi_cookie: u32,
    __bi_nr_segments: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkcg_gq {
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
struct bio_alloc_cache {
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct elevator_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_trace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_flush_queue {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_qos {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hd_geometry {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_report_zones_args {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_queue_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct disk_events {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct badblocks {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
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
struct blk_mq_ctx {
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
union __SlateRecord_blk_mq_h_125 {
    queuelist: list_head,
    rq_next: *mut request,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_mq_h_182 {
    hash: hlist_node,
    ipi_list: llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_blk_mq_h_193 {
    rb_node: rb_node,
    special_vec: bio_vec,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_202 {
    icq: *mut io_cq,
    r#priv: [*mut (); 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_207 {
    seq: u32,
    saved_end_io: Option<unsafe extern "C" fn(*mut request, u8, *const io_comp_batch) -> u32>,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct __SlateRecord_blk_mq_h_323 {
    lock: spinlock,
    dispatch: list_head,
    state: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits293U0,
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
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_388() };
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
struct scsi_cmnd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_target {
}


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_transport_template {
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
struct __SlateRecord_actypes_h_910 {
    r#type: u32,
    value: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_915 {
    r#type: u32,
    length: u32,
    pointer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_921 {
    r#type: u32,
    length: u32,
    pointer: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_927 {
    r#type: u32,
    count: u32,
    elements: *mut acpi_object,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_933 {
    r#type: u32,
    actual_type: u32,
    handle: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_939 {
    r#type: u32,
    proc_id: u32,
    pblk_address: u64,
    pblk_length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_actypes_h_946 {
    r#type: u32,
    system_level: u32,
    resource_order: u32,
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
union __SlateRecord_libata_h_586 {
    error: u8,
    feature: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_libata_h_597 {
    status: u8,
    command: u8,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
union __SlateRecord_libata_h_771 {
    id: [u16; 256],
    gscr: [u32; 128],
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
struct io_cq {
    q: *mut request_queue,
    ioc: *mut io_context,
    __slate_anon_2: __SlateRecord_iocontext_h_83,
    __slate_anon_3: __SlateRecord_iocontext_h_87,
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_context {
    refcount: __SlateRecord_types_h_197,
    active_ref: __SlateRecord_types_h_190,
    ioprio: u16,
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
    pub struct __SlateBits290U0 {
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
    pub struct __SlateBits293U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits805U0 {
        #[bits(30)]
        pub options: u32,
        #[bits(2)]
        pub mc_flags: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits805U1 {
        #[bits(2)]
        pub sanyo_slot: u8,
        #[bits(1)]
        pub keeplocked: u8,
        #[bits(5)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits497U0 {
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
    pub struct __SlateBits494U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits497U1 {
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
    pub struct __SlateBits497U2 {
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
    pub struct __SlateBits497U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits105U0 {
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
    pub struct __SlateBits124U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits629U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits416U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits470U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits472U0 {
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
    pub struct __SlateBits104U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits838U0 {
        #[bits(1)]
        pub tag_alloc_policy_rr: bool,
        #[bits(1)]
        pub track_queue_depth: u32,
        #[bits(2)]
        pub supported_mode: u32,
        #[bits(1)]
        pub emulated: u32,
        #[bits(1)]
        pub skip_settle_delay: u32,
        #[bits(1)]
        pub no_write_same: u32,
        #[bits(1)]
        pub host_tagset: u32,
        #[bits(1)]
        pub queuecommand_may_block: u32,
        #[bits(7, access = na)]
        pub __slate_pad_8: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits835U0 {
        #[bits(2)]
        pub active_mode: u32,
        #[bits(1)]
        pub host_self_blocked: u32,
        #[bits(1)]
        pub reverse_ordering: u32,
        #[bits(1)]
        pub tmf_in_progress: u32,
        #[bits(1)]
        pub no_write_same: u32,
        #[bits(1)]
        pub host_tagset: u32,
        #[bits(1)]
        pub queuecommand_may_block: u32,
        #[bits(1)]
        pub short_inquiry: u32,
        #[bits(1)]
        pub no_scsi2_lun_in_cdb: u32,
        #[bits(6, access = na)]
        pub __slate_pad_9: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits161U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits449U0 {
        #[bits(1)]
        pub is_added: u32,
        #[bits(1)]
        pub unsafe_warn: u32,
        #[bits(1)]
        pub flit_mode: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits736U0 {
        #[bits(1)]
        pub per_func_slot: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits731U0 {
        #[bits(3)]
        pub pcie_mpss: u8,
        #[bits(5, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits731U1 {
        #[bits(5)]
        pub pme_support: u32,
        #[bits(1)]
        pub pme_poll: u32,
        #[bits(1)]
        pub pinned: u32,
        #[bits(1)]
        pub config_rrs_sv: u32,
        #[bits(1)]
        pub imm_ready: u32,
        #[bits(1)]
        pub d1_support: u32,
        #[bits(1)]
        pub d2_support: u32,
        #[bits(1)]
        pub no_d1d2: u32,
        #[bits(1)]
        pub no_d3cold: u32,
        #[bits(1)]
        pub bridge_d3: u32,
        #[bits(1)]
        pub d3cold_allowed: u32,
        #[bits(1)]
        pub mmio_always_on: u32,
        #[bits(1)]
        pub wakeup_prepared: u32,
        #[bits(1)]
        pub skip_bus_pm: u32,
        #[bits(1)]
        pub ignore_hotplug: u32,
        #[bits(1)]
        pub hotplug_user_indicators: u32,
        #[bits(1)]
        pub clear_retrain_link: u32,
        #[bits(1)]
        pub no_bw_notif: u32,
        #[bits(2, access = na)]
        pub __slate_pad_18: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits731U2 {
        #[bits(1)]
        pub aspm_l0s_support: u32,
        #[bits(1)]
        pub aspm_l1_support: u32,
        #[bits(1)]
        pub ltr_path: u32,
        #[bits(1)]
        pub pasid_no_tlp: u32,
        #[bits(3)]
        pub eetlp_prefix_max: u32,
        #[bits(1, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield([u8; 6], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits731U3 {
        #[bits(1)]
        pub transparent: u32,
        #[bits(1)]
        pub io_window: u32,
        #[bits(1)]
        pub pref_window: u32,
        #[bits(1)]
        pub pref_64_window: u32,
        #[bits(1)]
        pub multifunction: u32,
        #[bits(1)]
        pub is_busmaster: u32,
        #[bits(1)]
        pub no_msi: u32,
        #[bits(1)]
        pub block_cfg_access: u32,
        #[bits(1)]
        pub broken_parity_status: u32,
        #[bits(2)]
        pub irq_reroute_variant: u32,
        #[bits(1)]
        pub msi_enabled: u32,
        #[bits(1)]
        pub msix_enabled: u32,
        #[bits(1)]
        pub ari_enabled: u32,
        #[bits(1)]
        pub ats_enabled: u32,
        #[bits(1)]
        pub pasid_enabled: u32,
        #[bits(1)]
        pub pri_enabled: u32,
        #[bits(1)]
        pub tph_enabled: u32,
        #[bits(1)]
        pub fm_enabled: u32,
        #[bits(1)]
        pub is_managed: u32,
        #[bits(1)]
        pub is_msi_managed: u32,
        #[bits(1)]
        pub needs_freset: u32,
        #[bits(1)]
        pub state_saved: u32,
        #[bits(1)]
        pub is_physfn: u32,
        #[bits(1)]
        pub is_virtfn: u32,
        #[bits(1)]
        pub is_hotplug_bridge: u32,
        #[bits(1)]
        pub is_pciehp: u32,
        #[bits(1)]
        pub shpc_managed: u32,
        #[bits(1)]
        pub is_thunderbolt: u32,
        #[bits(1)]
        pub is_cxl: u32,
        #[bits(1)]
        pub untrusted: u32,
        #[bits(1)]
        pub external_facing: u32,
        #[bits(1)]
        pub broken_intx_masking: u32,
        #[bits(1)]
        pub io_window_1k: u32,
        #[bits(1)]
        pub irq_managed: u32,
        #[bits(1)]
        pub non_compliant_bars: u32,
        #[bits(1)]
        pub is_probed: u32,
        #[bits(1)]
        pub link_active_reporting: u32,
        #[bits(1)]
        pub no_vf_scan: u32,
        #[bits(1)]
        pub no_command_memory: u32,
        #[bits(1)]
        pub rom_bar_overlap: u32,
        #[bits(1)]
        pub rom_attr_enabled: u32,
        #[bits(1)]
        pub non_mappable_bars: u32,
        #[bits(5, access = na)]
        pub __slate_pad_42: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits731U4 {
        #[bits(1)]
        pub pasid_required: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
unsafe { __compiletime_assert_409() };
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
    __slate_bits_0: __slate_bits::__SlateBits629U0,
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
    __slate_bits_0: __slate_bits::__SlateBits494U0,
}


fn devm_request_irq(dev: *mut device, irq: u32, handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, irqflags: u64, devname: *const u8, dev_id: *mut ()) -> i32 {
unsafe { devm_request_threaded_irq(dev, irq, handler, None, irqflags | 2097152, devname, dev_id) }
}

#[inline(always)]
fn __outb(mut value: u8, mut port: u16) {
unsafe {
core::arch::asm!("outb %al, %dx", in("ax") (value as u8) as u32, in("dx") port as u16, options(nostack, att_syntax));
}
}

#[inline(always)]
fn __inb(mut port: u16) -> u8 {
let mut value: u8 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("inb %dx, %al", lateout("ax") __t0, in("dx") port as u16, options(nostack, att_syntax));
value = unsafe { core::mem::transmute::<u8, u8>(__t0 as u8) };
}
value
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


// The PCI interface treats multi-function devices as independent
// devices.  The slot/function address of each device is encoded
// in a single byte as follows:
//
// 	7:3 = slot
// 	2:0 = function
//
// PCI_DEVFN(), PCI_SLOT(), and PCI_FUNC() are defined in uapi/linux/pci.h.
// In the interest of not exposing interfaces to user-space unnecessarily,
// the following kernel-only defines are being added here.
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_slot {
    bus: *mut pci_bus,
    list: list_head,
    hotplug: *mut hotplug_slot,
    number: u16,
    __slate_bits_0: __slate_bits::__SlateBits736U0,
    kobj: kobject,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_vpd {
    lock: mutex,
    len: u32,
    cap: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dev {
    bus_list: list_head,
    bus: *mut pci_bus,
    subordinate: *mut pci_bus,
    sysdata: *mut (),
    procent: *mut proc_dir_entry,
    slot: *mut pci_slot,
    devfn: u32,
    vendor: u16,
    device: u16,
    subsystem_vendor: u16,
    subsystem_device: u16,
    class: u32,
    revision: u8,
    hdr_type: u8,
    rcec_ea: *mut rcec_ea,
    rcec: *mut pci_dev,
    devcap: u32,
    rebar_cap: u16,
    pcie_cap: u8,
    msi_cap: u8,
    msix_cap: u8,
    __slate_bits_0: __slate_bits::__SlateBits731U0,
    rom_base_reg: u8,
    pin: u8,
    pcie_flags_reg: u16,
    dma_alias_mask: *mut u64,
    driver: *mut pci_driver,
    dma_mask: u64,
    msi_addr_mask: u64,
    dma_parms: device_dma_parameters,
    current_state: i32,
    pm_cap: u8,
    __slate_bits_1: __slate_bits::__SlateBits731U1,
    d3hot_delay: u32,
    d3cold_delay: u32,
    l1ss: u16,
    link_state: *mut pcie_link_state,
    __slate_bits_2: __slate_bits::__SlateBits731U2,
    error_state: u32,
    dev: device,
    cfg_size: i32,
    irq: u32,
    resource: [resource; 11],
    driver_exclusive_resource: resource,
    __slate_bits_3: __slate_bits::__SlateBits731U3,
    dev_flags: u16,
    enable_cnt: __SlateRecord_types_h_190,
    pcie_cap_lock: spinlock,
    saved_config_space: [u32; 16],
    saved_cap_space: hlist_head,
    msix_base: *mut (),
    msi_lock: raw_spinlock,
    vpd: pci_vpd,
    link_bwctrl: *mut pcie_bwctrl_data,
    __slate_anon_116: __SlateRecord_pci_h_547,
    ats_cap: u16,
    ats_stu: u8,
    pri_cap: u16,
    pri_reqs_alloc: u32,
    __slate_bits_4: __slate_bits::__SlateBits731U4,
    pasid_cap: u16,
    pasid_features: u16,
    acs_cap: u16,
    acs_capabilities: u16,
    supported_speeds: u8,
    rom: u64,
    romlen: u64,
    priv_flags: u64,
    reset_methods: [u8; 8],
    wake: *mut gpio_desc,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_bus {
    node: list_head,
    parent: *mut pci_bus,
    children: list_head,
    devices: list_head,
    self_: *mut pci_dev,
    slots: list_head,
    resource: [*mut resource; 4],
    resources: list_head,
    busn_res: resource,
    ops: *mut pci_ops,
    sysdata: *mut (),
    procdir: *mut proc_dir_entry,
    number: u8,
    primary: u8,
    max_bus_speed: u8,
    cur_bus_speed: u8,
    name: [u8; 48],
    bridge_ctl: u16,
    bus_flags: u16,
    bridge: *mut device,
    dev: device,
    __slate_bits_0: __slate_bits::__SlateBits449U0,
}


// Error values that may be returned by PCI functions
#[repr(C)]
#[derive(Clone, Copy)]
struct pci_ops {
    add_bus: Option<unsafe extern "C" fn(*mut pci_bus) -> i32>,
    remove_bus: Option<unsafe extern "C" fn(*mut pci_bus)>,
    map_bus: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32) -> *mut ()>,
    read: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, *mut u32) -> i32>,
    write: Option<unsafe extern "C" fn(*mut pci_bus, u32, i32, i32, u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_dynids {
    lock: spinlock,
    list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_error_handlers {
    error_detected: Option<unsafe extern "C" fn(*mut pci_dev, u32) -> u32>,
    mmio_enabled: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    slot_reset: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    reset_prepare: Option<unsafe extern "C" fn(*mut pci_dev)>,
    reset_done: Option<unsafe extern "C" fn(*mut pci_dev)>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev)>,
    cor_error_detected: Option<unsafe extern "C" fn(*mut pci_dev)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_driver {
    name: *const u8,
    id_table: *const pci_device_id,
    probe: Option<unsafe extern "C" fn(*mut pci_dev, *const pci_device_id) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut pci_dev)>,
    suspend: Option<unsafe extern "C" fn(*mut pci_dev, pm_message) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut pci_dev) -> i32>,
    shutdown: Option<unsafe extern "C" fn(*mut pci_dev)>,
    sriov_configure: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_set_msix_vec_count: Option<unsafe extern "C" fn(*mut pci_dev, i32) -> i32>,
    sriov_get_vf_total_msix: Option<unsafe extern "C" fn(*mut pci_dev) -> u32>,
    err_handler: *const pci_error_handlers,
    groups: *mut *const attribute_group,
    dev_groups: *mut *const attribute_group,
    driver: device_driver,
    dynids: pci_dynids,
    driver_managed_dma: bool,
}


// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
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
#[inline(always)]
fn lowmem_page_address(page: *const page) -> *mut () {
page_to_virt!(page)
}

// Mask used for checking in page_pool_page_is_pp() below. page->pp_magic is
// OR'ed with PP_SIGNATURE after the allocation in order to preserve bit 0 for
// the head page of compound page and bit 1 for pfmemalloc page, as well as the
// bits used for the DMA index. page_is_pfmemalloc() is checked in
// __page_pool_put_page() to avoid recycling the pfmemalloc page.
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

fn sg_chain_ptr(sg: *mut scatterlist) -> *mut scatterlist {
((unsafe { (*sg).page_link }) & !(1 | 2)) as *mut scatterlist
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

// Loop over each sg element, following the pointer to a new list if necessary
// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
fn dma_set_mask_and_coherent(dev: *mut device, mask: u64) -> i32 {
let rc: i32 = unsafe { dma_set_mask(dev, mask) };
if rc == 0 {
unsafe { dma_set_coherent_mask(dev, mask) };
}
rc
}

fn dmam_alloc_coherent(dev: *mut device, size: u64, dma_handle: *mut u64, gfp: u32) -> *mut () {
unsafe { dmam_alloc_attrs(dev, size, dma_handle, gfp, if gfp & (BIT!(13i32) as u32) != 0 { 256u64 } else { 0u64 }) }
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

#[inline(always)]
fn ndelay(nsec: u64) {
if unsafe { __slate_intrinsic_f217c4720f04f6db(nsec) } {
if nsec >= 20000 {
unsafe { __bad_ndelay() };
} else {
unsafe { __const_udelay(nsec.wrapping_mul(5u64)) };
}
} else {
unsafe { __ndelay(nsec) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ata_bmdma_prd {
    addr: u32,
    flags_len: u32,
}


fn ata_ok(status: u8) -> bool {
(status as i32) & (128 | 64 | 32 | 8 | 1) == 64
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


fn flush_dcache_page(page: *mut page) {
}

fn kmap_atomic(page: *const page) -> *mut () {
if IS_ENABLED!() != 0 {
migrate_disable();
} else {
loop {
preempt_count_inc!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
}
pagefault_disable();
page_address!(page)
}

fn __kunmap_atomic(addr: *const ()) {
pagefault_enable();
if IS_ENABLED!() != 0 {
migrate_enable();
} else {
loop {
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if unlikely!(preempt_count_dec_and_test!()) != 0 {
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
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_514: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

// kunmap_atomic - Unmap the virtual address mapped by kmap_atomic() - deprecated!
// @__addr:       Virtual address to be unmapped
//
// Unmaps an address previously mapped by kmap_atomic() and re-enables
// pagefaults. Depending on PREEMPT_RT configuration, re-enables also
// migration and preemption. Users should not count on these side effects.
//
// Mappings should be unmapped in the reverse order that they were mapped.
// See kmap_local_page() for details on nesting.
//
// @__addr can be any address within the mapped page, so there is no need
// to subtract any offset that has been added. In contrast to kunmap(),
// this function takes the address returned from kmap_atomic(), not the
// page passed to it. The compiler will warn you if you pass the page.
// kunmap_local - Unmap a page mapped via kmap_local_page().
// @__addr: An address within the page mapped
//
// @__addr can be any address within the mapped page.  Commonly it is the
// address return from kmap_local_page(), but it can also include offsets.
//
// Unmapping should be done in the reverse order of the mapping.  See
// kmap_local_page() for details.
#[repr(C)]
#[derive(Clone, Copy)]
struct bio_vec {
    bv_page: *mut page,
    bv_len: u32,
    bv_offset: u32,
}


#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct bvec_iter {
    bi_sector: u64,
    bi_size: u32,
    bi_idx: u32,
    bi_offset: u32,
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


// BLK_STS_AGAIN should only be returned if RQF_NOWAIT is set
// and the bio would block (cf bio_wouldblock_error())
#[repr(C)]
#[derive(Clone, Copy)]
struct bio {
    bi_next: *mut bio,
    bi_bdev: *mut block_device,
    bi_opf: u32,
    bi_flags: u16,
    bi_ioprio: u16,
    bi_write_hint: u8,
    bi_write_stream: u8,
    bi_status: u8,
    bi_bvec_gap_bit: u8,
    __bi_remaining: __SlateRecord_types_h_190,
    bi_io_vec: *mut bio_vec,
    bi_iter: bvec_iter,
    __slate_anon_12: __SlateRecord_blk_types_h_239,
    bi_end_io: Option<unsafe extern "C" fn(*mut bio)>,
    bi_private: *mut (),
    bi_blkg: *mut blkcg_gq,
    issue_time_ns: u64,
    bi_iocost_cost: u64,
    bi_vcnt: u16,
    bi_max_vecs: u16,
    __bi_cnt: __SlateRecord_types_h_190,
    bi_pool: *mut bio_set,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_list {
    head: *mut request,
    tail: *mut request,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_plug {
    mq_list: rq_list,
    cached_rqs: rq_list,
    cur_ktime: u64,
    nr_ios: u16,
    rq_count: u16,
    multiple_queues: bool,
    has_elevator: bool,
    cb_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mempool {
    lock: spinlock,
    min_nr: i32,
    curr_nr: i32,
    elements: *mut *mut (),
    pool_data: *mut (),
    alloc: Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>,
    free: Option<unsafe extern "C" fn(*mut (), *mut ())>,
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_list {
    head: *mut bio,
    tail: *mut bio,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct bio_set {
    bio_slab: *mut kmem_cache,
    front_pad: u32,
    cache: *mut bio_alloc_cache,
    bio_pool: mempool,
    bvec_pool: mempool,
    back_pad: u32,
    rescue_lock: spinlock,
    rescue_list: bio_list,
    rescue_work: work_struct,
    rescue_workqueue: *mut workqueue_struct,
    cpuhp_dead: hlist_node,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sbitmap_word {
    word: u64,
    __slate_pad_1: [u8; 56],
    cleared: u64,
    swap_lock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sbitmap {
    depth: u32,
    shift: u32,
    map_nr: u32,
    round_robin: bool,
    map: *mut sbitmap_word,
    alloc_hint: *mut u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct sbq_wait_state {
    wait: wait_queue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sbitmap_queue {
    sb: sbitmap,
    wake_batch: u32,
    wake_index: __SlateRecord_types_h_190,
    ws: *mut sbq_wait_state,
    ws_active: __SlateRecord_types_h_190,
    min_shallow_depth: u32,
    completion_cnt: __SlateRecord_types_h_190,
    wakeup_cnt: __SlateRecord_types_h_190,
}


// Enough for the string representation of any kind of UUID plus NULL.
// EFI UUID is 36 characters. MSDOS UUID is 11 characters.
#[repr(C)]
#[derive(Clone, Copy)]
struct partition_meta_info {
    uuid: [u8; 37],
    volname: [u8; 64],
}


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_port_ops: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_check_status: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_pause: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_dma_pause: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_wait_ready: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_dev_select: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_irq_on: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_tf_load: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_tf_read: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_exec_command: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_data_xfer: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_data_xfer32: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_hsm_move: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_queue_work: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_queue_delayed_work: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_queue_pio_task: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_qc_issue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_qc_fill_rtf: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_port_intr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_interrupt: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_lost_interrupt: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_freeze: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_thaw: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_prereset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_dev_classify: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_wait_after_reset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_softreset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_sata_sff_hardreset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_postreset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_drain_fifo: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_error_handler: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sff_std_ports: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_pci_sff_init_host: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_pci_sff_prepare_host: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_pci_sff_activate_host: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_pci_sff_init_one: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_port_ops: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma32_port_ops: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {37} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_qc_prep: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {38} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_dumb_qc_prep: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {39} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_qc_issue: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {40} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_port_intr: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {41} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_interrupt: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {42} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_error_handler: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {43} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_post_internal_cmd: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {44} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_irq_clear: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {45} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_setup: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {46} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_start: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {47} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_stop: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {48} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_status: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {49} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_port_start: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {50} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_bmdma_port_start32: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {51} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_pci_bmdma_clear_simplex: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {52} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_pci_bmdma_init: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {53} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_pci_bmdma_prepare_host: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {54} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_pci_bmdma_init_one: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {55} ; .previous", sym ata_sff_port_ops, sym ata_sff_check_status, sym ata_sff_pause, sym ata_sff_dma_pause, sym ata_sff_wait_ready, sym ata_sff_dev_select, sym ata_sff_irq_on, sym ata_sff_tf_load, sym ata_sff_tf_read, sym ata_sff_exec_command, sym ata_sff_data_xfer, sym ata_sff_data_xfer32, sym ata_sff_hsm_move, sym ata_sff_queue_work, sym ata_sff_queue_delayed_work, sym ata_sff_queue_pio_task, sym ata_sff_qc_issue, sym ata_sff_qc_fill_rtf, sym ata_sff_port_intr, sym ata_sff_interrupt, sym ata_sff_lost_interrupt, sym ata_sff_freeze, sym ata_sff_thaw, sym ata_sff_prereset, sym ata_sff_dev_classify, sym ata_sff_wait_after_reset, sym ata_sff_softreset, sym sata_sff_hardreset, sym ata_sff_postreset, sym ata_sff_drain_fifo, sym ata_sff_error_handler, sym ata_sff_std_ports, sym ata_pci_sff_init_host, sym ata_pci_sff_prepare_host, sym ata_pci_sff_activate_host, sym ata_pci_sff_init_one, sym ata_bmdma_port_ops, sym ata_bmdma32_port_ops, sym ata_bmdma_qc_prep, sym ata_bmdma_dumb_qc_prep, sym ata_bmdma_qc_issue, sym ata_bmdma_port_intr, sym ata_bmdma_interrupt, sym ata_bmdma_error_handler, sym ata_bmdma_post_internal_cmd, sym ata_bmdma_irq_clear, sym ata_bmdma_setup, sym ata_bmdma_start, sym ata_bmdma_stop, sym ata_bmdma_status, sym ata_bmdma_port_start, sym ata_bmdma_port_start32, sym ata_pci_bmdma_clear_simplex, sym ata_pci_bmdma_init, sym ata_pci_bmdma_prepare_host, sym ata_pci_bmdma_init_one, options(att_syntax));

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_integrity {
    flags: u8,
    csum_type: u8,
    metadata_size: u8,
    pi_offset: u8,
    interval_exp: u8,
    tag_size: u8,
    pi_tuple_size: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
    major: i32,
    first_minor: i32,
    minors: i32,
    disk_name: [u8; 32],
    events: u16,
    event_flags: u16,
    part_tbl: xarray,
    part0: *mut block_device,
    fops: *const block_device_operations,
    queue: *mut request_queue,
    private_data: *mut (),
    bio_split: bio_set,
    flags: i32,
    state: u64,
    open_mutex: mutex,
    open_partitions: u32,
    bdi: *mut backing_dev_info,
    queue_kobj: kobject,
    slave_dir: *mut kobject,
    slave_bdevs: list_head,
    random: *mut timer_rand_state,
    ev: *mut disk_events,
    cdi: *mut cdrom_device_info,
    node_id: i32,
    bb: *mut badblocks,
    lockdep_map: lockdep_map,
    diskseq: u64,
    open_mode: u32,
    ia_ranges: *mut blk_independent_access_ranges,
    rqos_state_mutex: mutex,
}


// supports a volatile write cache
// supports passing on the FUA bit
// don't modify data until writeback is done
// always completes in submit context
// supports REQ_NOWAIT
// is a zoned device
// do not send FLUSH/FUA commands despite advertising a write cache
#[repr(C)]
#[derive(Clone, Copy)]
struct queue_limits {
    features: u32,
    flags: u32,
    seg_boundary_mask: u64,
    virt_boundary_mask: u64,
    max_hw_sectors: u32,
    max_dev_sectors: u32,
    chunk_sectors: u32,
    max_sectors: u32,
    max_user_sectors: u32,
    max_segment_size: u32,
    max_fast_segment_size: u32,
    physical_block_size: u32,
    logical_block_size: u32,
    alignment_offset: u32,
    io_min: u32,
    io_opt: u32,
    max_discard_sectors: u32,
    max_hw_discard_sectors: u32,
    max_user_discard_sectors: u32,
    max_secure_erase_sectors: u32,
    max_write_zeroes_sectors: u32,
    max_wzeroes_unmap_sectors: u32,
    max_hw_wzeroes_unmap_sectors: u32,
    max_user_wzeroes_unmap_sectors: u32,
    max_hw_zone_append_sectors: u32,
    max_zone_append_sectors: u32,
    discard_granularity: u32,
    discard_alignment: u32,
    zone_write_granularity: u32,
    atomic_write_hw_max: u32,
    atomic_write_max_sectors: u32,
    atomic_write_hw_boundary: u32,
    atomic_write_boundary_sectors: u32,
    atomic_write_hw_unit_min: u32,
    atomic_write_unit_min: u32,
    atomic_write_hw_unit_max: u32,
    atomic_write_unit_max: u32,
    max_segments: u16,
    max_integrity_segments: u16,
    max_discard_segments: u16,
    max_write_streams: u16,
    write_stream_granularity: u32,
    max_open_zones: u32,
    max_active_zones: u32,
    dma_alignment: u32,
    dma_pad_mask: u32,
    integrity: blk_integrity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_independent_access_range {
    kobj: kobject,
    sector: u64,
    nr_sectors: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_independent_access_ranges {
    kobj: kobject,
    sysfs_registered: bool,
    nr_ia_ranges: u32,
    ia_range: [blk_independent_access_range; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct request_queue {
    queuedata: *mut (),
    elevator: *mut elevator_queue,
    mq_ops: *const blk_mq_ops,
    queue_ctx: *mut blk_mq_ctx,
    queue_flags: u64,
    rq_timeout: u32,
    queue_depth: u32,
    refs: refcount_struct,
    nr_hw_queues: u32,
    queue_hw_ctx: *mut *mut blk_mq_hw_ctx,
    q_usage_counter: percpu_ref,
    io_lock_cls_key: lock_class_key,
    io_lockdep_map: lockdep_map,
    q_lock_cls_key: lock_class_key,
    q_lockdep_map: lockdep_map,
    last_merge: *mut request,
    queue_lock: spinlock,
    quiesce_depth: i32,
    disk: *mut gendisk,
    mq_kobj: *mut kobject,
    limits: queue_limits,
    dev: *mut device,
    rpm_status: i32,
    pm_only: __SlateRecord_types_h_190,
    stats: *mut blk_queue_stats,
    rq_qos: *mut rq_qos,
    rq_qos_mutex: mutex,
    id: i32,
    nr_requests: u32,
    async_depth: u32,
    timeout: timer_list,
    timeout_work: work_struct,
    nr_active_requests_shared_tags: __SlateRecord_types_h_190,
    sched_shared_tags: *mut blk_mq_tags,
    icq_list: list_head,
    blkcg_pols: [u64; 1],
    root_blkg: *mut blkcg_gq,
    blkg_list: list_head,
    blkcg_mutex: mutex,
    node: i32,
    requeue_lock: spinlock,
    requeue_list: list_head,
    requeue_work: delayed_work,
    blk_trace: *mut blk_trace,
    fq: *mut blk_flush_queue,
    flush_list: list_head,
    elevator_lock: mutex,
    sysfs_lock: mutex,
    limits_lock: mutex,
    unused_hctx_list: list_head,
    unused_hctx_lock: spinlock,
    mq_freeze_depth: i32,
    callback_head: callback_head,
    mq_freeze_wq: wait_queue_head,
    mq_freeze_lock: mutex,
    tag_set: *mut blk_mq_tag_set,
    tag_set_list: list_head,
    debugfs_dir: *mut dentry,
    sched_debugfs_dir: *mut dentry,
    rqos_debugfs_dir: *mut dentry,
    debugfs_mutex: mutex,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device_operations {
    submit_bio: Option<unsafe extern "C" fn(*mut bio)>,
    poll_bio: Option<unsafe extern "C" fn(*mut bio, *mut io_comp_batch, u32) -> i32>,
    open: Option<unsafe extern "C" fn(*mut gendisk, u32) -> i32>,
    release: Option<unsafe extern "C" fn(*mut gendisk)>,
    ioctl: Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>,
    check_events: Option<unsafe extern "C" fn(*mut gendisk, u32) -> u32>,
    unlock_native_capacity: Option<unsafe extern "C" fn(*mut gendisk)>,
    getgeo: Option<unsafe extern "C" fn(*mut gendisk, *mut hd_geometry) -> i32>,
    set_read_only: Option<unsafe extern "C" fn(*mut block_device, bool) -> i32>,
    free_disk: Option<unsafe extern "C" fn(*mut gendisk)>,
    swap_slot_free_notify: Option<unsafe extern "C" fn(*mut block_device, u64)>,
    report_zones: Option<unsafe extern "C" fn(*mut gendisk, u64, u32, *mut blk_report_zones_args) -> i32>,
    devnode: Option<unsafe extern "C" fn(*mut gendisk, *mut u16) -> *mut u8>,
    get_unique_id: Option<unsafe extern "C" fn(*mut gendisk, *mut u8, u32) -> i32>,
    owner: *mut module,
    pr_ops: *const pr_ops,
    alternative_gpt_sector: Option<unsafe extern "C" fn(*mut gendisk, *mut u64) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_holder_ops {
    mark_dead: Option<unsafe extern "C" fn(*mut block_device, bool)>,
    sync: Option<unsafe extern "C" fn(*mut block_device)>,
    freeze: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
    thaw: Option<unsafe extern "C" fn(*mut block_device) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_comp_batch {
    req_list: rq_list,
    need_ts: bool,
    complete: Option<unsafe extern "C" fn(*mut io_comp_batch)>,
    poll_ctx: *mut (),
}


// Maximum contiguous integrity buffer allocation.
#[repr(C)]
#[derive(Clone, Copy)]
struct request {
    q: *mut request_queue,
    mq_ctx: *mut blk_mq_ctx,
    mq_hctx: *mut blk_mq_hw_ctx,
    cmd_flags: u32,
    rq_flags: u32,
    tag: i32,
    internal_tag: i32,
    timeout: u32,
    __data_len: u32,
    __sector: u64,
    bio: *mut bio,
    biotail: *mut bio,
    __slate_anon_12: __SlateRecord_blk_mq_h_125,
    part: *mut block_device,
    alloc_time_ns: u64,
    start_time_ns: u64,
    io_start_time_ns: u64,
    stats_sectors: u16,
    nr_phys_segments: u16,
    nr_integrity_segments: u16,
    phys_gap_bit: u8,
    state: u32,
    r#ref: __SlateRecord_types_h_190,
    deadline: u64,
    __slate_anon_24: __SlateRecord_blk_mq_h_182,
    __slate_anon_25: __SlateRecord_blk_mq_h_193,
    elv: __SlateRecord_blk_mq_h_202,
    flush: __SlateRecord_blk_mq_h_207,
    fifo_time: u64,
    end_io: Option<unsafe extern "C" fn(*mut request, u8, *const io_comp_batch) -> u32>,
    end_io_data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_hw_ctx {
    __slate_anon_0: __SlateRecord_blk_mq_h_323,
    run_work: delayed_work,
    cpumask: [cpumask; 1],
    next_cpu: i32,
    next_cpu_batch: i32,
    flags: u64,
    sched_data: *mut (),
    queue: *mut request_queue,
    fq: *mut blk_flush_queue,
    driver_data: *mut (),
    ctx_map: sbitmap,
    dispatch_from: *mut blk_mq_ctx,
    dispatch_busy: u32,
    r#type: u16,
    nr_ctx: u16,
    ctxs: *mut *mut blk_mq_ctx,
    dispatch_wait_lock: spinlock,
    dispatch_wait: wait_queue_entry,
    wait_index: __SlateRecord_types_h_190,
    tags: *mut blk_mq_tags,
    sched_tags: *mut blk_mq_tags,
    numa_node: i32,
    queue_num: u32,
    nr_active: __SlateRecord_types_h_190,
    cpuhp_online: hlist_node,
    cpuhp_dead: hlist_node,
    kobj: kobject,
    debugfs_dir: *mut dentry,
    sched_debugfs_dir: *mut dentry,
    hctx_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_queue_map {
    mq_map: *mut u32,
    nr_queues: u32,
    queue_offset: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tag_set {
    ops: *const blk_mq_ops,
    map: [blk_mq_queue_map; 3],
    nr_maps: u32,
    nr_hw_queues: u32,
    queue_depth: u32,
    reserved_tags: u32,
    cmd_size: u32,
    numa_node: i32,
    timeout: u32,
    flags: u32,
    driver_data: *mut (),
    tags: *mut *mut blk_mq_tags,
    shared_tags: *mut blk_mq_tags,
    tag_list_lock: mutex,
    tag_list: list_head,
    srcu: *mut srcu_struct,
    tags_srcu: srcu_struct,
    update_nr_hwq_lock: rw_semaphore,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_queue_data {
    rq: *mut request,
    last: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ops {
    queue_rq: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *const blk_mq_queue_data) -> u8>,
    commit_rqs: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx)>,
    queue_rqs: Option<unsafe extern "C" fn(*mut rq_list)>,
    get_budget: Option<unsafe extern "C" fn(*mut request_queue) -> i32>,
    put_budget: Option<unsafe extern "C" fn(*mut request_queue, i32)>,
    set_rq_budget_token: Option<unsafe extern "C" fn(*mut request, i32)>,
    get_rq_budget_token: Option<unsafe extern "C" fn(*mut request) -> i32>,
    timeout: Option<unsafe extern "C" fn(*mut request) -> u32>,
    poll: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *mut io_comp_batch) -> i32>,
    complete: Option<unsafe extern "C" fn(*mut request)>,
    init_hctx: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, *mut (), u32) -> i32>,
    exit_hctx: Option<unsafe extern "C" fn(*mut blk_mq_hw_ctx, u32)>,
    init_request: Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut request, u32, i32) -> i32>,
    exit_request: Option<unsafe extern "C" fn(*mut blk_mq_tag_set, *mut request, u32)>,
    cleanup_rq: Option<unsafe extern "C" fn(*mut request)>,
    busy: Option<unsafe extern "C" fn(*mut request_queue) -> bool>,
    map_queues: Option<unsafe extern "C" fn(*mut blk_mq_tag_set)>,
    show_rq: Option<unsafe extern "C" fn(*mut seq_file, *mut request)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_tags {
    nr_tags: u32,
    nr_reserved_tags: u32,
    active_queues: u32,
    bitmap_tags: sbitmap_queue,
    breserved_tags: sbitmap_queue,
    rqs: *mut *mut request,
    static_rqs: *mut *mut request,
    page_list: list_head,
    lock: spinlock,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_sense_hdr {
    response_code: u8,
    sense_key: u8,
    asc: u8,
    ascq: u8,
    byte4: u8,
    byte5: u8,
    byte6: u8,
    additional_length: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_host_template {
    cmd_size: u32,
    queuecommand: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> u32>,
    queue_reserved_command: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> u32>,
    commit_rqs: Option<unsafe extern "C" fn(*mut Scsi_Host, u16)>,
    module: *mut module,
    name: *const u8,
    info: Option<unsafe extern "C" fn(*mut Scsi_Host) -> *const u8>,
    ioctl: Option<unsafe extern "C" fn(*mut scsi_device, u32, *mut ()) -> i32>,
    compat_ioctl: Option<unsafe extern "C" fn(*mut scsi_device, u32, *mut ()) -> i32>,
    init_cmd_priv: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> i32>,
    exit_cmd_priv: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> i32>,
    eh_abort_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_device_reset_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_target_reset_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_bus_reset_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_host_reset_handler: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    sdev_init: Option<unsafe extern "C" fn(*mut scsi_device) -> i32>,
    sdev_configure: Option<unsafe extern "C" fn(*mut scsi_device, *mut queue_limits) -> i32>,
    sdev_destroy: Option<unsafe extern "C" fn(*mut scsi_device)>,
    target_alloc: Option<unsafe extern "C" fn(*mut scsi_target) -> i32>,
    target_destroy: Option<unsafe extern "C" fn(*mut scsi_target)>,
    scan_finished: Option<unsafe extern "C" fn(*mut Scsi_Host, u64) -> i32>,
    scan_start: Option<unsafe extern "C" fn(*mut Scsi_Host)>,
    change_queue_depth: Option<unsafe extern "C" fn(*mut scsi_device, i32) -> i32>,
    map_queues: Option<unsafe extern "C" fn(*mut Scsi_Host)>,
    mq_poll: Option<unsafe extern "C" fn(*mut Scsi_Host, u32) -> i32>,
    dma_need_drain: Option<unsafe extern "C" fn(*mut request) -> bool>,
    bios_param: Option<unsafe extern "C" fn(*mut scsi_device, *mut gendisk, u64, *mut i32) -> i32>,
    unlock_native_capacity: Option<unsafe extern "C" fn(*mut scsi_device)>,
    show_info: Option<unsafe extern "C" fn(*mut seq_file, *mut Scsi_Host) -> i32>,
    write_info: Option<unsafe extern "C" fn(*mut Scsi_Host, *mut u8, i32) -> i32>,
    eh_timed_out: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> u32>,
    eh_should_retry_cmd: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> bool>,
    host_reset: Option<unsafe extern "C" fn(*mut Scsi_Host, i32) -> i32>,
    proc_name: *const u8,
    can_queue: i32,
    nr_reserved_cmds: i32,
    this_id: i32,
    sg_tablesize: u16,
    sg_prot_tablesize: u16,
    max_sectors: u32,
    max_segment_size: u32,
    dma_alignment: u32,
    dma_boundary: u64,
    virt_boundary_mask: u64,
    cmd_per_lun: i16,
    __slate_bits_0: __slate_bits::__SlateBits838U0,
    max_host_blocked: u32,
    shost_groups: *mut *const attribute_group,
    sdev_groups: *mut *const attribute_group,
    vendor_id: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct Scsi_Host {
    __devices: list_head,
    __targets: list_head,
    starved_list: list_head,
    default_lock: spinlock,
    host_lock: *mut spinlock,
    scan_mutex: mutex,
    eh_abort_list: list_head,
    eh_cmd_q: list_head,
    ehandler: *mut task_struct,
    eh_action: *mut completion,
    host_wait: wait_queue_head,
    hostt: *const scsi_host_template,
    transportt: *mut scsi_transport_template,
    tagset_refcnt: kref,
    tagset_freed: completion,
    tag_set: blk_mq_tag_set,
    host_blocked: __SlateRecord_types_h_190,
    host_failed: u32,
    host_eh_scheduled: u32,
    host_no: u32,
    eh_deadline: i32,
    last_reset: u64,
    max_channel: u32,
    max_id: u32,
    max_lun: u64,
    unique_id: u32,
    max_cmd_len: u16,
    this_id: i32,
    can_queue: i32,
    nr_reserved_cmds: u32,
    cmd_per_lun: i16,
    sg_tablesize: u16,
    sg_prot_tablesize: u16,
    max_sectors: u32,
    opt_sectors: u32,
    max_segment_size: u32,
    dma_alignment: u32,
    dma_boundary: u64,
    virt_boundary_mask: u64,
    nr_hw_queues: u32,
    nr_maps: u32,
    async_scan: bool,
    eh_noresume: bool,
    __slate_bits_0: __slate_bits::__SlateBits835U0,
    work_q: *mut workqueue_struct,
    tmf_work_q: *mut workqueue_struct,
    max_host_blocked: u32,
    prot_capabilities: u32,
    prot_guard_type: u8,
    base: u64,
    io_port: u64,
    n_io_port: u8,
    dma_channel: u8,
    irq: u32,
    shost_state: u32,
    shost_gendev: device,
    shost_dev: device,
    pseudo_sdev: *mut scsi_device,
    shost_data: *mut (),
    dma_dev: *mut device,
    eh_work: work_struct,
    rpm_autosuspend_delay: i32,
    hostdata: [u64; 0],
}


// Fixed by ACPI spec
#[repr(C)]
#[derive(Clone, Copy)]
union acpi_object {
    r#type: u32,
    integer: __SlateRecord_actypes_h_910,
    string: __SlateRecord_actypes_h_915,
    buffer: __SlateRecord_actypes_h_921,
    package: __SlateRecord_actypes_h_927,
    reference: __SlateRecord_actypes_h_933,
    processor: __SlateRecord_actypes_h_939,
    power_resource: __SlateRecord_actypes_h_946,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_msf0 {
    minute: u8,
    second: u8,
    frame: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union cdrom_addr {
    msf: cdrom_msf0,
    lba: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_multisession {
    addr: cdrom_addr,
    xa_flag: u8,
    addr_format: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_mcn {
    medium_catalog_number: [u8; 14],
}


// seconds per minute
// frames per second
// MSF numbering offset of first frame
#[repr(C)]
#[derive(Clone, Copy)]
struct packet_command {
    cmd: [u8; 12],
    buffer: *mut u8,
    buflen: u32,
    stat: i32,
    sshdr: *mut scsi_sense_hdr,
    data_direction: u8,
    quiet: i32,
    timeout: i32,
    reserved: [*mut (); 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
    ops: *const cdrom_device_ops,
    list: list_head,
    disk: *mut gendisk,
    handle: *mut (),
    mask: i32,
    speed: i32,
    capacity: i32,
    __slate_bits_0: __slate_bits::__SlateBits805U0,
    vfs_events: u32,
    ioctl_events: u32,
    use_count: i32,
    name: [u8; 20],
    __slate_bits_1: __slate_bits::__SlateBits805U1,
    cdda_method: i32,
    last_sense: u8,
    media_written: u8,
    mmc3_profile: u16,
    mrw_mode_page: i32,
    opened_for_data: bool,
    last_media_change_ms: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_ops {
    open: Option<unsafe extern "C" fn(*mut cdrom_device_info, i32) -> i32>,
    release: Option<unsafe extern "C" fn(*mut cdrom_device_info)>,
    drive_status: Option<unsafe extern "C" fn(*mut cdrom_device_info, i32) -> i32>,
    check_events: Option<unsafe extern "C" fn(*mut cdrom_device_info, u32, i32) -> u32>,
    tray_move: Option<unsafe extern "C" fn(*mut cdrom_device_info, i32) -> i32>,
    lock_door: Option<unsafe extern "C" fn(*mut cdrom_device_info, i32) -> i32>,
    select_speed: Option<unsafe extern "C" fn(*mut cdrom_device_info, u64) -> i32>,
    get_last_session: Option<unsafe extern "C" fn(*mut cdrom_device_info, *mut cdrom_multisession) -> i32>,
    get_mcn: Option<unsafe extern "C" fn(*mut cdrom_device_info, *mut cdrom_mcn) -> i32>,
    reset: Option<unsafe extern "C" fn(*mut cdrom_device_info) -> i32>,
    audio_ioctl: Option<unsafe extern "C" fn(*mut cdrom_device_info, u32, *mut ()) -> i32>,
    generic_packet: Option<unsafe extern "C" fn(*mut cdrom_device_info, *mut packet_command) -> i32>,
    read_cdda_bpc: Option<unsafe extern "C" fn(*mut cdrom_device_info, *mut (), u32, u32, *mut u8) -> i32>,
    capability: i32,
}


// The SCSI spec says there could be 256 slots.
#[repr(C)]
#[derive(Clone, Copy)]
struct ata_taskfile {
    flags: u64,
    protocol: u8,
    ctl: u8,
    hob_feature: u8,
    hob_nsect: u8,
    hob_lbal: u8,
    hob_lbam: u8,
    hob_lbah: u8,
    __slate_anon_8: __SlateRecord_libata_h_586,
    nsect: u8,
    lbal: u8,
    lbam: u8,
    lbah: u8,
    device: u8,
    __slate_anon_14: __SlateRecord_libata_h_597,
    auxiliary: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_ioports {
    cmd_addr: *mut (),
    data_addr: *mut (),
    error_addr: *mut (),
    feature_addr: *mut (),
    nsect_addr: *mut (),
    lbal_addr: *mut (),
    lbam_addr: *mut (),
    lbah_addr: *mut (),
    device_addr: *mut (),
    status_addr: *mut (),
    command_addr: *mut (),
    altstatus_addr: *mut (),
    ctl_addr: *mut (),
    bmdma_addr: *mut (),
    scr_addr: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_host {
    lock: spinlock,
    dev: *mut device,
    iomap: *const *mut (),
    n_ports: u32,
    n_tags: u32,
    private_data: *mut (),
    ops: *mut ata_port_operations,
    flags: u64,
    kref: kref,
    eh_mutex: mutex,
    eh_owner: *mut task_struct,
    simplex_claimed: *mut ata_port,
    ports: [*mut ata_port; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_queued_cmd {
    ap: *mut ata_port,
    dev: *mut ata_device,
    scsicmd: *mut scsi_cmnd,
    scsidone: Option<unsafe extern "C" fn(*mut scsi_cmnd)>,
    tf: ata_taskfile,
    cdb: [u8; 16],
    flags: u64,
    tag: u32,
    hw_tag: u32,
    n_elem: u32,
    orig_n_elem: u32,
    dma_dir: i32,
    sect_size: u32,
    nbytes: u32,
    extrabytes: u32,
    curbytes: u32,
    sgent: scatterlist,
    sg: *mut scatterlist,
    cursg: *mut scatterlist,
    cursg_ofs: u32,
    err_mask: u32,
    result_tf: ata_taskfile,
    complete_fn: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    private_data: *mut (),
    lldd_task: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_port_stats {
    unhandled_irq: u64,
    idle_irq: u64,
    rw_reqbuf: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_ering_entry {
    eflags: u32,
    err_mask: u32,
    timestamp: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_ering {
    cursor: i32,
    ring: [ata_ering_entry; 32],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_cpr {
    num: u8,
    num_storage_elements: u8,
    start_lba: u64,
    num_lbas: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_cpr_log {
    nr_cpr: u8,
    cpr: [ata_cpr; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_cdl {
    desc_log_buf: [u8; 512],
    ncq_sense_log_buf: [u8; 1024],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_device {
    link: *mut ata_link,
    devno: u32,
    quirks: u64,
    flags: u64,
    sdev: *mut scsi_device,
    private_data: *mut (),
    gtf_cache: *mut acpi_object,
    gtf_filter: u32,
    tdev: device,
    n_sectors: u64,
    n_native_sectors: u64,
    class: u32,
    unpark_deadline: u64,
    pio_mode: u8,
    dma_mode: u8,
    xfer_mode: u8,
    xfer_shift: u32,
    multi_count: u32,
    max_sectors: u32,
    cdb_len: u32,
    pio_mask: u32,
    mwdma_mask: u32,
    udma_mask: u32,
    cylinders: u16,
    heads: u16,
    sectors: u16,
    __slate_anon_26: __SlateRecord_libata_h_771,
    gp_log_dir: [u8; 512],
    devslp_timing: [u8; 8],
    ncq_send_recv_cmds: [u8; 20],
    ncq_non_data_cmds: [u8; 64],
    zac_zoned_cap: u32,
    zac_zones_optimal_open: u32,
    zac_zones_optimal_nonseq: u32,
    zac_zones_max_open: u32,
    cpr_log: *mut ata_cpr_log,
    cdl: *mut ata_cdl,
    spdn_cnt: i32,
    ering: ata_ering,
    __slate_pad_39: [u8; 48],
    sector_buf: [u8; 512],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_eh_info {
    dev: *mut ata_device,
    serror: u32,
    err_mask: u32,
    action: u32,
    dev_action: [u32; 2],
    flags: u32,
    probe_mask: u32,
    desc: [u8; 80],
    desc_len: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_eh_context {
    i: ata_eh_info,
    tries: [i32; 2],
    cmd_timeout_idx: [[i32; 9]; 2],
    classes: [u32; 2],
    did_probe_mask: u32,
    unloaded_mask: u32,
    saved_ncq_enabled: u32,
    saved_xfer_mode: [u8; 2],
    last_reset: u64,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ata_acpi_drive {
    pio: u32,
    dma: u32,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct ata_acpi_gtm {
    drive: [ata_acpi_drive; 2],
    flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_link {
    ap: *mut ata_port,
    pmp: i32,
    tdev: device,
    active_tag: u32,
    sactive: u32,
    flags: u32,
    saved_scontrol: u32,
    hw_sata_spd_limit: u32,
    sata_spd_limit: u32,
    sata_spd: u32,
    lpm_policy: u32,
    deferred_qc_work: work_struct,
    deferred_qc: *mut ata_queued_cmd,
    eh_info: ata_eh_info,
    eh_context: ata_eh_context,
    device: [ata_device; 2],
    last_lpm_change: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_port {
    scsi_host: *mut Scsi_Host,
    ops: *mut ata_port_operations,
    lock: *mut spinlock,
    flags: u64,
    pflags: u32,
    print_id: u32,
    port_no: u32,
    ioaddr: ata_ioports,
    ctl: u8,
    last_ctl: u8,
    sff_pio_task_link: *mut ata_link,
    sff_pio_task: delayed_work,
    bmdma_prd: *mut ata_bmdma_prd,
    bmdma_prd_dma: u64,
    pio_mask: u32,
    mwdma_mask: u32,
    udma_mask: u32,
    cbl: u32,
    qcmd: [ata_queued_cmd; 33],
    qc_active: u64,
    nr_active_links: i32,
    link: ata_link,
    slave_link: *mut ata_link,
    nr_pmp_links: i32,
    pmp_link: *mut ata_link,
    excl_link: *mut ata_link,
    stats: ata_port_stats,
    host: *mut ata_host,
    dev: *mut device,
    tdev: device,
    scsi_scan_mutex: mutex,
    hotplug_task: delayed_work,
    scsi_rescan_task: delayed_work,
    hsm_task_state: u32,
    eh_done_q: list_head,
    eh_wait_q: wait_queue_head,
    eh_tries: i32,
    park_req_pending: completion,
    pm_mesg: pm_message,
    target_lpm_policy: u32,
    fastdrain_timer: timer_list,
    fastdrain_cnt: u32,
    cookie: u64,
    em_message_type: i32,
    private_data: *mut (),
    __acpi_init_gtm: ata_acpi_gtm,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_reset_operations {
    prereset: Option<unsafe extern "C" fn(*mut ata_link, u64) -> i32>,
    softreset: Option<unsafe extern "C" fn(*mut ata_link, *mut u32, u64) -> i32>,
    hardreset: Option<unsafe extern "C" fn(*mut ata_link, *mut u32, u64) -> i32>,
    postreset: Option<unsafe extern "C" fn(*mut ata_link, *mut u32)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_port_operations {
    qc_defer: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> i32>,
    check_atapi_dma: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> i32>,
    qc_prep: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>,
    qc_issue: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>,
    qc_fill_rtf: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    qc_ncq_fill_rtf: Option<unsafe extern "C" fn(*mut ata_port, u64)>,
    cable_detect: Option<unsafe extern "C" fn(*mut ata_port) -> i32>,
    mode_filter: Option<unsafe extern "C" fn(*mut ata_device, u32) -> u32>,
    set_piomode: Option<unsafe extern "C" fn(*mut ata_port, *mut ata_device)>,
    set_dmamode: Option<unsafe extern "C" fn(*mut ata_port, *mut ata_device)>,
    set_mode: Option<unsafe extern "C" fn(*mut ata_link, *mut *mut ata_device) -> i32>,
    read_id: Option<unsafe extern "C" fn(*mut ata_device, *mut ata_taskfile, *mut u16) -> u32>,
    dev_config: Option<unsafe extern "C" fn(*mut ata_device)>,
    freeze: Option<unsafe extern "C" fn(*mut ata_port)>,
    thaw: Option<unsafe extern "C" fn(*mut ata_port)>,
    reset: ata_reset_operations,
    pmp_reset: ata_reset_operations,
    error_handler: Option<unsafe extern "C" fn(*mut ata_port)>,
    lost_interrupt: Option<unsafe extern "C" fn(*mut ata_port)>,
    post_internal_cmd: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    sched_eh: Option<unsafe extern "C" fn(*mut ata_port)>,
    end_eh: Option<unsafe extern "C" fn(*mut ata_port)>,
    scr_read: Option<unsafe extern "C" fn(*mut ata_link, u32, *mut u32) -> i32>,
    scr_write: Option<unsafe extern "C" fn(*mut ata_link, u32, u32) -> i32>,
    pmp_attach: Option<unsafe extern "C" fn(*mut ata_port)>,
    pmp_detach: Option<unsafe extern "C" fn(*mut ata_port)>,
    set_lpm: Option<unsafe extern "C" fn(*mut ata_link, u32, u32) -> i32>,
    port_suspend: Option<unsafe extern "C" fn(*mut ata_port, pm_message) -> i32>,
    port_resume: Option<unsafe extern "C" fn(*mut ata_port) -> i32>,
    port_start: Option<unsafe extern "C" fn(*mut ata_port) -> i32>,
    port_stop: Option<unsafe extern "C" fn(*mut ata_port)>,
    host_stop: Option<unsafe extern "C" fn(*mut ata_host)>,
    sff_dev_select: Option<unsafe extern "C" fn(*mut ata_port, u32)>,
    sff_set_devctl: Option<unsafe extern "C" fn(*mut ata_port, u8)>,
    sff_check_status: Option<unsafe extern "C" fn(*mut ata_port) -> u8>,
    sff_check_altstatus: Option<unsafe extern "C" fn(*mut ata_port) -> u8>,
    sff_tf_load: Option<unsafe extern "C" fn(*mut ata_port, *const ata_taskfile)>,
    sff_tf_read: Option<unsafe extern "C" fn(*mut ata_port, *mut ata_taskfile)>,
    sff_exec_command: Option<unsafe extern "C" fn(*mut ata_port, *const ata_taskfile)>,
    sff_data_xfer: Option<unsafe extern "C" fn(*mut ata_queued_cmd, *mut u8, u32, i32) -> u32>,
    sff_irq_on: Option<unsafe extern "C" fn(*mut ata_port)>,
    sff_irq_check: Option<unsafe extern "C" fn(*mut ata_port) -> bool>,
    sff_irq_clear: Option<unsafe extern "C" fn(*mut ata_port)>,
    sff_drain_fifo: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    bmdma_setup: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    bmdma_start: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    bmdma_stop: Option<unsafe extern "C" fn(*mut ata_queued_cmd)>,
    bmdma_status: Option<unsafe extern "C" fn(*mut ata_port) -> u8>,
    em_show: Option<unsafe extern "C" fn(*mut ata_port, *mut u8) -> i64>,
    em_store: Option<unsafe extern "C" fn(*mut ata_port, *const u8, u64) -> i64>,
    sw_activity_show: Option<unsafe extern "C" fn(*mut ata_device, *mut u8) -> i64>,
    sw_activity_store: Option<unsafe extern "C" fn(*mut ata_device, u32) -> i64>,
    transmit_led_message: Option<unsafe extern "C" fn(*mut ata_port, u32, i64) -> i64>,
    inherits: *const ata_port_operations,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_port_info {
    flags: u64,
    link_flags: u64,
    pio_mask: u32,
    mwdma_mask: u32,
    udma_mask: u32,
    port_ops: *mut ata_port_operations,
    private_data: *mut (),
}


fn ata_is_atapi(prot: u8) -> bool {
(prot as i32) & 8 != 0
}

fn ata_is_dma(prot: u8) -> bool {
(prot as i32) & 2 != 0
}

fn is_multi_taskfile(tf: *mut ata_taskfile) -> i32 {
(((unsafe { (*tf).__slate_anon_14.command }) as i32) == 196 || ((unsafe { (*tf).__slate_anon_14.command }) as i32) == 197 || ((unsafe { (*tf).__slate_anon_14.command }) as i32) == 41 || ((unsafe { (*tf).__slate_anon_14.command }) as i32) == 57 || ((unsafe { (*tf).__slate_anon_14.command }) as i32) == 206) as i32
}

fn ata_port_is_dummy(ap: *mut ata_port) -> i32 {
((unsafe { (*ap).ops }) == unsafe { core::ptr::addr_of_mut!(ata_dummy_port_ops) }) as i32
}

fn sata_ehc_deb_timing(ehc: *mut ata_eh_context) -> *const u32 {
if (unsafe { (*ehc).i.flags }) & 1 != 0 {
return unsafe { core::ptr::addr_of!(sata_deb_timing_hotplug) as *const u32 };
} else {
return unsafe { core::ptr::addr_of!(sata_deb_timing_normal) as *const u32 };
}
}

fn sata_pmp_attached(ap: *mut ata_port) -> bool {
(unsafe { (*ap).nr_pmp_links }) != 0
}

fn ata_port_desc_misc(ap: *mut ata_port, irq: i32) {
unsafe { ata_port_desc(ap, c"irq %d".as_ptr() as *const u8, irq) };
unsafe { ata_port_desc(ap, c"lpm-pol %d".as_ptr() as *const u8, (*ap).target_lpm_policy as u32) };
if (unsafe { (*ap).pflags }) & 4194304 != 0 {
unsafe { ata_port_desc(ap, c"ext".as_ptr() as *const u8) };
}
}

fn ata_tag_internal(tag: u32) -> bool {
tag == 32
}

fn ata_tag_valid(tag: u32) -> bool {
tag < 32 || ata_tag_internal(tag)
}

fn ata_qc_set_polling(__v0: *mut ata_queued_cmd) {
unsafe {
(*__v0).tf.ctl = (((*__v0).tf.ctl as i32) | 2) as u8;
}
}

fn __ata_qc_from_tag(ap: *mut ata_port, tag: u32) -> *mut ata_queued_cmd {
if ata_tag_valid(tag) {
return unsafe { (*ap).qcmd.as_mut_ptr() as *mut ata_queued_cmd }.wrapping_offset(tag as isize);
}
core::ptr::null_mut::<ata_queued_cmd>()
}

fn ata_qc_from_tag(ap: *mut ata_port, tag: u32) -> *mut ata_queued_cmd {
let qc: *mut ata_queued_cmd = __ata_qc_from_tag(ap, tag);
if unlikely!(qc.is_null()) != 0 {
return qc;
}
if (unsafe { (*qc).flags }) & 65537u64 == 1 {
return qc;
}
core::ptr::null_mut::<ata_queued_cmd>()
}

fn ac_err_mask(status: u8) -> u32 {
if (status as i32) & (128 | 8) != 0 {
return 2;
}
if (status as i32) & (1 | 32) != 0 {
return 1;
}
0
}

fn __ac_err_mask(status: u8) -> u32 {
let mask: u32 = ac_err_mask(status);
if mask == 0 {
return 256;
}
mask
}

fn ata_check_ready(status: u8) -> i32 {
if (status as i32) & 128 == 0 {
return 1;
}
// 0xff indicates either no device or device not ready
if (status as i32) == 255 {
return 19i32.wrapping_neg();
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_dma_pause")]
extern "C" fn ata_sff_dma_pause(ap: *mut ata_port) {
// An altstatus read will cause the needed delay without
// messing up the IRQ status
if ata_sff_altstatus(ap, core::ptr::null_mut::<u8>()) {
return;
}
// There are no DMA controllers without ctl. BUG here to ensure
// we never violate the HDMA1:0 transition timing and risk
// corruption.
loop {
unsafe {
core::arch::asm!("849: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 849b - ., 3; .popsection\n# {0}", const 849 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ata_sff_dma_pause_0, sym __slate_asm_string_ata_sff_dma_pause_1, const 182 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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

fn ata_sff_busy_wait(ap: *mut ata_port, bits: u32, mut max: u32) -> u8 {
let mut status: u8 = 0;
loop {
udelay(10u64);
status = unsafe { unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap) };
max = max.wrapping_sub(1);
if !((status as i32) != 255 && ((status as i32) as u32) & bits != 0 && max > 0) {
break;
}
}
status
}

fn ata_wait_idle(ap: *mut ata_port) -> u8 {
let status: u8 = ata_sff_busy_wait(ap, 136u32, 1000u32);
if (status as i32) != 255 && (status as i32) & (128 | 8) != 0 {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017ata%u: abnormal Status 0x%X\n".as_ptr() as *const u8, (*ap).print_id, status as i32) };
}
__v0 = 0;
}
status
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
fn __do_trace_ata_tf_load(ap: *mut ata_port, tf: *const ata_taskfile) {
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
let mut __UNIQUE_ID_guard_651: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_651) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_652: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_652) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_654() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_tf_load)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_653: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_653;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile) -> i32>>(__SCT__tp_func_ata_tf_load as *const ()) };
unsafe { __v12.unwrap()(__data, ap, tf) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_tf_load: [u8; 12] = [97, 116, 97, 95, 116, 102, 95, 108, 111, 97, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_tf_load_655: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_tf_load) }) as *mut ();

fn trace_ata_tf_load(ap: *mut ata_port, tf: *const ata_taskfile) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_tf_load.key) }) != 0 {
__do_trace_ata_tf_load(ap, tf);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_656: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_exec_command(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
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
let mut __UNIQUE_ID_guard_658: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_658) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_659: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_659) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_661() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_exec_command)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_660: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_660;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile, u32) -> i32>>(__SCT__tp_func_ata_exec_command as *const ()) };
unsafe { __v12.unwrap()(__data, ap, tf, tag) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_exec_command: __SlateAlign16<[u8; 17]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 101u8, 120u8, 101u8, 99u8, 95u8, 99u8, 111u8, 109u8, 109u8, 97u8, 110u8, 100u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_exec_command_662: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_exec_command) }) as *mut ();

fn trace_ata_exec_command(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_exec_command.key) }) != 0 {
__do_trace_ata_exec_command(ap, tf, tag);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_663: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_bmdma_setup(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
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
let mut __UNIQUE_ID_guard_665: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_665) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_666: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_666) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_668() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_bmdma_setup)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_667: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_667;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile, u32) -> i32>>(__SCT__tp_func_ata_bmdma_setup as *const ()) };
unsafe { __v12.unwrap()(__data, ap, tf, tag) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_bmdma_setup: __SlateAlign16<[u8; 16]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 98u8, 109u8, 100u8, 109u8, 97u8, 95u8, 115u8, 101u8, 116u8, 117u8, 112u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_bmdma_setup_669: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_bmdma_setup) }) as *mut ();

fn trace_ata_bmdma_setup(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_bmdma_setup.key) }) != 0 {
__do_trace_ata_bmdma_setup(ap, tf, tag);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_670: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_bmdma_start(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
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
let mut __UNIQUE_ID_guard_672: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_672) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_673: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_673) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_675() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_bmdma_start)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_674: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_674;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile, u32) -> i32>>(__SCT__tp_func_ata_bmdma_start as *const ()) };
unsafe { __v12.unwrap()(__data, ap, tf, tag) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_bmdma_start: __SlateAlign16<[u8; 16]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 98u8, 109u8, 100u8, 109u8, 97u8, 95u8, 115u8, 116u8, 97u8, 114u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_bmdma_start_676: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_bmdma_start) }) as *mut ();

fn trace_ata_bmdma_start(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
if static_branch_unlikely__4!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_bmdma_start.key) }) != 0 {
__do_trace_ata_bmdma_start(ap, tf, tag);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_677: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_bmdma_stop(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
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
let mut __UNIQUE_ID_guard_679: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_679) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_680: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_680) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_682() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_bmdma_stop)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_681: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_681;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_port, *const ata_taskfile, u32) -> i32>>(__SCT__tp_func_ata_bmdma_stop as *const ()) };
unsafe { __v12.unwrap()(__data, ap, tf, tag) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_bmdma_stop: [u8; 15] = [97, 116, 97, 95, 98, 109, 100, 109, 97, 95, 115, 116, 111, 112, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_bmdma_stop_683: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_bmdma_stop) }) as *mut ();

fn trace_ata_bmdma_stop(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
if static_branch_unlikely__5!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_bmdma_stop.key) }) != 0 {
__do_trace_ata_bmdma_stop(ap, tf, tag);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_684: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_bmdma_status(ap: *mut ata_port, host_stat: u32) {
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
let mut __UNIQUE_ID_guard_686: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_686) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_687: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_687) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_689() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_bmdma_status)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_688: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_688;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_port, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_port, u32) -> i32>>(__SCT__tp_func_ata_bmdma_status as *const ()) };
unsafe { __v12.unwrap()(__data, ap, host_stat) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_bmdma_status: __SlateAlign16<[u8; 17]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 98u8, 109u8, 100u8, 109u8, 97u8, 95u8, 115u8, 116u8, 97u8, 116u8, 117u8, 115u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_bmdma_status_690: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_bmdma_status) }) as *mut ();

fn trace_ata_bmdma_status(ap: *mut ata_port, host_stat: u32) {
if static_branch_unlikely__6!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_bmdma_status.key) }) != 0 {
__do_trace_ata_bmdma_status(ap, host_stat);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_691: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_sff_hsm_state(qc: *mut ata_queued_cmd, state: u8) {
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
let mut __UNIQUE_ID_guard_798: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_798) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_799: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_799) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_801() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_sff_hsm_state)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_800: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_800;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u8) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u8) -> i32>>(__SCT__tp_func_ata_sff_hsm_state as *const ()) };
unsafe { __v12.unwrap()(__data, qc, state) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_sff_hsm_state: __SlateAlign16<[u8; 18]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 115u8, 102u8, 102u8, 95u8, 104u8, 115u8, 109u8, 95u8, 115u8, 116u8, 97u8, 116u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_sff_hsm_state_802: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_sff_hsm_state) }) as *mut ();

fn trace_ata_sff_hsm_state(qc: *mut ata_queued_cmd, state: u8) {
if static_branch_unlikely__7!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_sff_hsm_state.key) }) != 0 {
__do_trace_ata_sff_hsm_state(qc, state);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_803: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_sff_hsm_command_complete(qc: *mut ata_queued_cmd, state: u8) {
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
let mut __UNIQUE_ID_guard_805: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_805) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_806: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_806) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_808() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_sff_hsm_command_complete)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_807: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_807;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u8) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u8) -> i32>>(__SCT__tp_func_ata_sff_hsm_command_complete as *const ()) };
unsafe { __v12.unwrap()(__data, qc, state) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_sff_hsm_command_complete: __SlateAlign16<[u8; 29]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 115u8, 102u8, 102u8, 95u8, 104u8, 115u8, 109u8, 95u8, 99u8, 111u8, 109u8, 109u8, 97u8, 110u8, 100u8, 95u8, 99u8, 111u8, 109u8, 112u8, 108u8, 101u8, 116u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_sff_hsm_command_complete_809: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_sff_hsm_command_complete) }) as *mut ();

fn trace_ata_sff_hsm_command_complete(qc: *mut ata_queued_cmd, state: u8) {
if static_branch_unlikely__12!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_sff_hsm_command_complete.key) }) != 0 {
__do_trace_ata_sff_hsm_command_complete(qc, state);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_810: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_sff_port_intr(qc: *mut ata_queued_cmd, state: u8) {
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
let mut __UNIQUE_ID_guard_812: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_812) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_813: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_813) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_815() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_sff_port_intr)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_814: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_814;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u8) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u8) -> i32>>(__SCT__tp_func_ata_sff_port_intr as *const ()) };
unsafe { __v12.unwrap()(__data, qc, state) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_sff_port_intr: __SlateAlign16<[u8; 18]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 115u8, 102u8, 102u8, 95u8, 112u8, 111u8, 114u8, 116u8, 95u8, 105u8, 110u8, 116u8, 114u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_sff_port_intr_816: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_sff_port_intr) }) as *mut ();

fn trace_ata_sff_port_intr(qc: *mut ata_queued_cmd, state: u8) {
if static_branch_unlikely__8!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_sff_port_intr.key) }) != 0 {
__do_trace_ata_sff_port_intr(qc, state);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_817: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_sff_pio_transfer_data(qc: *mut ata_queued_cmd, offset: u32, count: u32) {
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
let mut __UNIQUE_ID_guard_819: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_819) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_820: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_820) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_822() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_sff_pio_transfer_data)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_821: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_821;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u32, u32) -> i32>>(__SCT__tp_func_ata_sff_pio_transfer_data as *const ()) };
unsafe { __v12.unwrap()(__data, qc, offset, count) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_sff_pio_transfer_data: __SlateAlign16<[u8; 26]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 115u8, 102u8, 102u8, 95u8, 112u8, 105u8, 111u8, 95u8, 116u8, 114u8, 97u8, 110u8, 115u8, 102u8, 101u8, 114u8, 95u8, 100u8, 97u8, 116u8, 97u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_sff_pio_transfer_data_823: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_sff_pio_transfer_data) }) as *mut ();

fn trace_ata_sff_pio_transfer_data(qc: *mut ata_queued_cmd, offset: u32, count: u32) {
if static_branch_unlikely__9!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_sff_pio_transfer_data.key) }) != 0 {
__do_trace_ata_sff_pio_transfer_data(qc, offset, count);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_824: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_atapi_pio_transfer_data(qc: *mut ata_queued_cmd, offset: u32, count: u32) {
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
let mut __UNIQUE_ID_guard_826: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_826) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_827: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_827) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_829() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_atapi_pio_transfer_data)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_828: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_828;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u32, u32) -> i32>>(__SCT__tp_func_atapi_pio_transfer_data as *const ()) };
unsafe { __v12.unwrap()(__data, qc, offset, count) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_atapi_pio_transfer_data: __SlateAlign16<[u8; 24]> = __SlateAlign16([97u8, 116u8, 97u8, 112u8, 105u8, 95u8, 112u8, 105u8, 111u8, 95u8, 116u8, 114u8, 97u8, 110u8, 115u8, 102u8, 101u8, 114u8, 95u8, 100u8, 97u8, 116u8, 97u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_atapi_pio_transfer_data_830: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_atapi_pio_transfer_data) }) as *mut ();

fn trace_atapi_pio_transfer_data(qc: *mut ata_queued_cmd, offset: u32, count: u32) {
if static_branch_unlikely__13!(unsafe { core::ptr::addr_of_mut!(__tracepoint_atapi_pio_transfer_data.key) }) != 0 {
__do_trace_atapi_pio_transfer_data(qc, offset, count);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_831: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_atapi_send_cdb(qc: *mut ata_queued_cmd, offset: u32, count: u32) {
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
let mut __UNIQUE_ID_guard_833: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_833) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_834: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_834) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_836() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_atapi_send_cdb)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_835: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_835;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u32, u32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_queued_cmd, u32, u32) -> i32>>(__SCT__tp_func_atapi_send_cdb as *const ()) };
unsafe { __v12.unwrap()(__data, qc, offset, count) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_atapi_send_cdb: [u8; 15] = [97, 116, 97, 112, 105, 95, 115, 101, 110, 100, 95, 99, 100, 98, 0];

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_atapi_send_cdb_837: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_atapi_send_cdb) }) as *mut ();

fn trace_atapi_send_cdb(qc: *mut ata_queued_cmd, offset: u32, count: u32) {
if static_branch_unlikely__10!(unsafe { core::ptr::addr_of_mut!(__tracepoint_atapi_send_cdb.key) }) != 0 {
__do_trace_atapi_send_cdb(qc, offset, count);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_838: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_ata_sff_flush_pio_task(ap: *mut ata_port) {
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
let mut __UNIQUE_ID_guard_840: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_840) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_841: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_841) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_843() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_ata_sff_flush_pio_task)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_842: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_842;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut ata_port) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ata_port) -> i32>>(__SCT__tp_func_ata_sff_flush_pio_task as *const ()) };
unsafe { __v12.unwrap()(__data, ap) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_ata_sff_flush_pio_task: __SlateAlign16<[u8; 23]> = __SlateAlign16([97u8, 116u8, 97u8, 95u8, 115u8, 102u8, 102u8, 95u8, 102u8, 108u8, 117u8, 115u8, 104u8, 95u8, 112u8, 105u8, 111u8, 95u8, 116u8, 97u8, 115u8, 107u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_ata_sff_flush_pio_task_844: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_ata_sff_flush_pio_task) }) as *mut ();

fn trace_ata_sff_flush_pio_task(ap: *mut ata_port) {
if static_branch_unlikely__11!(unsafe { core::ptr::addr_of_mut!(__tracepoint_ata_sff_flush_pio_task.key) }) != 0 {
__do_trace_ata_sff_flush_pio_task(ap);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_845: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

static mut ata_sff_wq: *mut workqueue_struct = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static ata_sff_port_ops: __SlateConst<ata_port_operations> = __SlateConst({
let mut __t0: ata_port_operations = unsafe { core::mem::zeroed() };
__t0.qc_issue = Some(ata_sff_qc_issue);
__t0.qc_fill_rtf = Some(ata_sff_qc_fill_rtf);
__t0.freeze = Some(ata_sff_freeze);
__t0.thaw = Some(ata_sff_thaw);
__t0.reset = ata_reset_operations { prereset: Some(ata_sff_prereset), softreset: Some(ata_sff_softreset), hardreset: Some(sata_sff_hardreset), postreset: Some(ata_sff_postreset) };
__t0.error_handler = Some(ata_sff_error_handler);
__t0.lost_interrupt = Some(ata_sff_lost_interrupt);
__t0.sff_dev_select = Some(ata_sff_dev_select);
__t0.sff_check_status = Some(ata_sff_check_status);
__t0.sff_tf_load = Some(ata_sff_tf_load);
__t0.sff_tf_read = Some(ata_sff_tf_read);
__t0.sff_exec_command = Some(ata_sff_exec_command);
__t0.sff_data_xfer = Some(ata_sff_data_xfer);
__t0.sff_drain_fifo = Some(ata_sff_drain_fifo);
__t0.inherits = unsafe { core::ptr::addr_of!(ata_base_port_ops) };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_port_ops_846: *mut () = (unsafe { core::ptr::addr_of!(ata_sff_port_ops.0) }) as *mut ();

/// ata_sff_check_status - Read device status reg & clear interrupt
/// @ap: port where the device is
///
/// Reads ATA taskfile status register for currently-selected device
/// and return its value. This also clears pending interrupts
///     from this device
///
/// LOCKING:
/// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_check_status")]
extern "C" fn ata_sff_check_status(ap: *mut ata_port) -> u8 {
(unsafe { ioread8((*ap).ioaddr.status_addr as *const ()) }) as u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_check_status_847: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port) -> u8>, *mut ()>(Some(ata_sff_check_status)) };

/// ata_sff_altstatus - Read device alternate status reg
/// @ap: port where the device is
/// @status: pointer to a status value
///
/// Reads ATA alternate status register for currently-selected device
/// and return its value.
///
/// RETURN:
/// true if the register exists, false if not.
///
/// LOCKING:
/// Inherited from caller.
fn ata_sff_altstatus(ap: *mut ata_port, status: *mut u8) -> bool {
let mut tmp: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u8 = core::ptr::addr_of_mut!(tmp) as *mut u8;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if (*(*ap).ops).sff_check_altstatus != None {
*__slate_slot_0 = unsafe { (*(*ap).ops).sff_check_altstatus }.unwrap()(ap);
} else {
if !unsafe { (*ap).ioaddr.altstatus_addr }.is_null() {
*__slate_slot_0 = ioread8((*ap).ioaddr.altstatus_addr as *const ()) as u8;
} else {
return false;
}
}
if !status.is_null() {
*status = *__slate_slot_0;
}
return true;
}
}

/// ata_sff_irq_status - Check if the device is busy
/// @ap: port where the device is
///
/// Determine if the port is currently busy. Uses altstatus
/// if available in order to avoid clearing shared IRQ status
/// when finding an IRQ source. Non ctl capable devices don't
/// share interrupt lines fortunately for us.
///
/// LOCKING:
/// Inherited from caller.
fn ata_sff_irq_status(ap: *mut ata_port) -> u8 {
let mut status: u8 = 0;
// Not us: We are busy
if ata_sff_altstatus(ap, core::ptr::addr_of_mut!(status)) && (status as i32) & 128 != 0 {
return status;
}
// Clear INTRQ latch
status = unsafe { unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap) };
status
}

// ata_sff_sync - Flush writes
// @ap: Port to wait for.
//
// CAUTION:
// If we have an mmio device with no ctl and no altstatus
// method this will fail. No such devices are known to exist.
//
// LOCKING:
// Inherited from caller.
fn ata_sff_sync(ap: *mut ata_port) {
ata_sff_altstatus(ap, core::ptr::null_mut::<u8>());
}

// ata_sff_pause		-	Flush writes and wait 400nS
// @ap: Port to pause for.
//
// CAUTION:
// If we have an mmio device with no ctl and no altstatus
// method this will fail. No such devices are known to exist.
//
// LOCKING:
// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_pause")]
extern "C" fn ata_sff_pause(ap: *mut ata_port) {
ata_sff_sync(ap);
ndelay!(400i32);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_pause_848: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_sff_pause)) };

// ata_sff_dma_pause	-	Pause before commencing DMA
// @ap: Port to pause for.
//
// Perform I/O fencing and ensure sufficient cycle delays occur
// for the HDMA1:0 transition
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_dma_pause_850: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_sff_dma_pause)) };

#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_check_ready")]
extern "C" fn ata_sff_check_ready(link: *mut ata_link) -> i32 {
let status: u8 = unsafe { unsafe { (*(*(*link).ap).ops).sff_check_status }.unwrap()((*link).ap) };
ata_check_ready(status)
}

/// ata_sff_wait_ready - sleep until BSY clears, or timeout
/// @link: SFF link to wait ready status for
/// @deadline: deadline jiffies for the operation
///
/// Sleep until ATA Status register bit BSY clears, or timeout
/// occurs.
///
/// LOCKING:
/// Kernel thread context (may sleep).
///
/// RETURNS:
/// 0 on success, -errno otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_wait_ready")]
extern "C" fn ata_sff_wait_ready(link: *mut ata_link, deadline: u64) -> i32 {
unsafe { ata_wait_ready(link, deadline, Some(ata_sff_check_ready)) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_wait_ready_851: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, u64) -> i32>, *mut ()>(Some(ata_sff_wait_ready)) };

/// ata_sff_set_devctl - Write device control reg
/// @ap: port where the device is
/// @ctl: value to write
///
/// Writes ATA device control register.
///
/// RETURN:
/// true if the register exists, false if not.
///
/// LOCKING:
/// Inherited from caller.
fn ata_sff_set_devctl(ap: *mut ata_port, ctl: u8) -> bool {
if (unsafe { (*(*ap).ops).sff_set_devctl }) != None {
unsafe { unsafe { (*(*ap).ops).sff_set_devctl }.unwrap()(ap, ctl) };
return true;
}
if !unsafe { (*ap).ioaddr.ctl_addr }.is_null() {
unsafe { iowrite8(ctl, (*ap).ioaddr.ctl_addr) };
return true;
}
false
}

/// ata_sff_dev_select - Select device 0/1 on ATA bus
/// @ap: ATA channel to manipulate
/// @device: ATA device (numbered from zero) to select
///
/// Use the method defined in the ATA specification to
/// make either device 0, or device 1, active on the
/// ATA channel.  Works with both PIO and MMIO.
///
/// May be used as the dev_select() entry in ata_port_operations.
///
/// LOCKING:
/// caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_dev_select")]
extern "C" fn ata_sff_dev_select(ap: *mut ata_port, device: u32) {
let tmp: u8 = if device == 0 {
160
} else {
176u8
};
unsafe { iowrite8(tmp, (*ap).ioaddr.device_addr) };
ata_sff_pause(ap); // needed; also flushes, for mmio
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_dev_select_852: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, u32)>, *mut ()>(Some(ata_sff_dev_select)) };

/// ata_dev_select - Select device 0/1 on ATA bus
/// @ap: ATA channel to manipulate
/// @device: ATA device (numbered from zero) to select
/// @wait: non-zero to wait for Status register BSY bit to clear
/// @can_sleep: non-zero if context allows sleeping
///
/// Use the method defined in the ATA specification to
/// make either device 0, or device 1, active on the
/// ATA channel.
///
/// This is a high-level version of ata_sff_dev_select(), which
/// additionally provides the services of inserting the proper
/// pauses and status polling, where needed.
///
/// LOCKING:
/// caller.
fn ata_dev_select(ap: *mut ata_port, device: u32, wait: u32, can_sleep: u32) {
if wait != 0 {
ata_wait_idle(ap);
}
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, device) };
if wait != 0 {
if can_sleep != 0 && (unsafe { (*((*ap).link.device.as_mut_ptr() as *mut ata_device).wrapping_offset(device as isize)).class }) == 3 {
unsafe { ata_msleep(ap, 150u32) };
}
ata_wait_idle(ap);
}
}

/// ata_sff_irq_on - Enable interrupts on a port.
/// @ap: Port on which interrupts are enabled.
///
/// Enable interrupts on a legacy IDE device using MMIO or PIO,
/// wait for idle, clear any pending interrupts.
///
/// Note: may NOT be used as the sff_irq_on() entry in
/// ata_port_operations.
///
/// LOCKING:
/// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_irq_on")]
extern "C" fn ata_sff_irq_on(ap: *mut ata_port) {
if (unsafe { (*(*ap).ops).sff_irq_on }) != None {
unsafe { unsafe { (*(*ap).ops).sff_irq_on }.unwrap()(ap) };
return;
}
let __v2: u8 = (((unsafe { (*ap).ctl }) as i32) & !2) as u8;
unsafe {
(*ap).ctl = __v2;
}
unsafe {
(*ap).last_ctl = (*ap).ctl;
}
ata_sff_set_devctl(ap, unsafe { (*ap).ctl });
ata_wait_idle(ap);
if (unsafe { (*(*ap).ops).sff_irq_clear }) != None {
unsafe { unsafe { (*(*ap).ops).sff_irq_clear }.unwrap()(ap) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_irq_on_853: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_sff_irq_on)) };

/// ata_sff_tf_load - send taskfile registers to host controller
/// @ap: Port to which output is sent
/// @tf: ATA taskfile register set
///
/// Outputs ATA taskfile to standard ATA host controller.
///
/// LOCKING:
/// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_tf_load")]
extern "C" fn ata_sff_tf_load(ap: *mut ata_port, tf: *const ata_taskfile) {
let ioaddr: *mut ata_ioports = unsafe { core::ptr::addr_of_mut!((*ap).ioaddr) };
let is_addr: u32 = ((unsafe { (*tf).flags }) & 2) as u32;
if ((unsafe { (*tf).ctl }) as i32) != ((unsafe { (*ap).last_ctl }) as i32) {
if !unsafe { (*ioaddr).ctl_addr }.is_null() {
unsafe { iowrite8((*tf).ctl, (*ioaddr).ctl_addr) };
}
unsafe {
(*ap).last_ctl = (*tf).ctl;
}
ata_wait_idle(ap);
}
if is_addr != 0 && (unsafe { (*tf).flags }) & 1 != 0 {
let mut __v0: i64;
let __ret_warn_on: i32 = unsafe { (*ioaddr).ctl_addr }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("854: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 854b - ., 3; .popsection\n# {0}", const 854 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_sff_tf_load_1, sym __slate_asm_string_ata_sff_tf_load_2, const 355 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("855: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 855b - ., 4; .popsection\n# {0}", const 855 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { iowrite8((*tf).hob_feature, (*ioaddr).feature_addr) };
unsafe { iowrite8((*tf).hob_nsect, (*ioaddr).nsect_addr) };
unsafe { iowrite8((*tf).hob_lbal, (*ioaddr).lbal_addr) };
unsafe { iowrite8((*tf).hob_lbam, (*ioaddr).lbam_addr) };
unsafe { iowrite8((*tf).hob_lbah, (*ioaddr).lbah_addr) };
}
if is_addr != 0 {
unsafe { iowrite8((*tf).__slate_anon_8.feature, (*ioaddr).feature_addr) };
unsafe { iowrite8((*tf).nsect, (*ioaddr).nsect_addr) };
unsafe { iowrite8((*tf).lbal, (*ioaddr).lbal_addr) };
unsafe { iowrite8((*tf).lbam, (*ioaddr).lbam_addr) };
unsafe { iowrite8((*tf).lbah, (*ioaddr).lbah_addr) };
}
if (unsafe { (*tf).flags }) & 4 != 0 {
unsafe { iowrite8((*tf).device, (*ioaddr).device_addr) };
}
ata_wait_idle(ap);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_tf_load_856: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, *const ata_taskfile)>, *mut ()>(Some(ata_sff_tf_load)) };

/// ata_sff_tf_read - input device's ATA taskfile shadow registers
/// @ap: Port from which input is read
/// @tf: ATA taskfile register set for storing input
///
/// Reads ATA taskfile registers for currently-selected device
/// into @tf. Assumes the device has a fully SFF compliant task file
/// layout and behaviour. If you device does not (eg has a different
/// status method) then you will need to provide a replacement tf_read
///
/// LOCKING:
/// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_tf_read")]
extern "C" fn ata_sff_tf_read(ap: *mut ata_port, tf: *mut ata_taskfile) {
let ioaddr: *mut ata_ioports = unsafe { core::ptr::addr_of_mut!((*ap).ioaddr) };
unsafe {
(*tf).__slate_anon_14.status = ata_sff_check_status(ap);
}
unsafe {
(*tf).__slate_anon_8.error = ioread8((*ioaddr).error_addr as *const ()) as u8;
}
unsafe {
(*tf).nsect = ioread8((*ioaddr).nsect_addr as *const ()) as u8;
}
unsafe {
(*tf).lbal = ioread8((*ioaddr).lbal_addr as *const ()) as u8;
}
unsafe {
(*tf).lbam = ioread8((*ioaddr).lbam_addr as *const ()) as u8;
}
unsafe {
(*tf).lbah = ioread8((*ioaddr).lbah_addr as *const ()) as u8;
}
unsafe {
(*tf).device = ioread8((*ioaddr).device_addr as *const ()) as u8;
}
if (unsafe { (*tf).flags }) & 1 != 0 {
if likely__6!(unsafe { (*ioaddr).ctl_addr }) != 0 {
unsafe { iowrite8((((*tf).ctl as i32) | 128) as u8, (*ioaddr).ctl_addr) };
unsafe {
(*tf).hob_feature = ioread8((*ioaddr).error_addr as *const ()) as u8;
}
unsafe {
(*tf).hob_nsect = ioread8((*ioaddr).nsect_addr as *const ()) as u8;
}
unsafe {
(*tf).hob_lbal = ioread8((*ioaddr).lbal_addr as *const ()) as u8;
}
unsafe {
(*tf).hob_lbam = ioread8((*ioaddr).lbam_addr as *const ()) as u8;
}
unsafe {
(*tf).hob_lbah = ioread8((*ioaddr).lbah_addr as *const ()) as u8;
}
unsafe { iowrite8((*tf).ctl, (*ioaddr).ctl_addr) };
unsafe {
(*ap).last_ctl = (*tf).ctl;
}
} else {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("857: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 857b - ., 3; .popsection\n# {0}", const 857 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_sff_tf_read_1, sym __slate_asm_string_ata_sff_tf_read_2, const 414 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("858: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 858b - ., 4; .popsection\n# {0}", const 858 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_tf_read_859: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, *mut ata_taskfile)>, *mut ()>(Some(ata_sff_tf_read)) };

/// ata_sff_exec_command - issue ATA command to host controller
/// @ap: port to which command is being issued
/// @tf: ATA taskfile register set
///
/// Issues ATA command, with proper synchronization with interrupt
/// handler / other threads.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_exec_command")]
extern "C" fn ata_sff_exec_command(ap: *mut ata_port, tf: *const ata_taskfile) {
unsafe { iowrite8((*tf).__slate_anon_14.command, (*ap).ioaddr.command_addr) };
ata_sff_pause(ap);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_exec_command_860: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, *const ata_taskfile)>, *mut ()>(Some(ata_sff_exec_command)) };

/// ata_tf_to_host - issue ATA taskfile to host controller
/// @ap: port to which command is being issued
/// @tf: ATA taskfile register set
/// @tag: tag of the associated command
///
/// Issues ATA taskfile register set to ATA host controller,
/// with proper synchronization with interrupt handler and
/// other threads.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_tf_to_host(ap: *mut ata_port, tf: *const ata_taskfile, tag: u32) {
trace_ata_tf_load(ap, tf);
unsafe { unsafe { (*(*ap).ops).sff_tf_load }.unwrap()(ap, tf) };
trace_ata_exec_command(ap, tf, tag);
unsafe { unsafe { (*(*ap).ops).sff_exec_command }.unwrap()(ap, tf) };
}

/// ata_sff_data_xfer - Transfer data by PIO
/// @qc: queued command
/// @buf: data buffer
/// @buflen: buffer length
/// @rw: read/write
///
/// Transfer data from/to the device data register by PIO.
///
/// LOCKING:
/// Inherited from caller.
///
/// RETURNS:
/// Bytes consumed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_data_xfer")]
extern "C" fn ata_sff_data_xfer(qc: *mut ata_queued_cmd, mut buf: *mut u8, buflen: u32, rw: i32) -> u32 {
let ap: *mut ata_port = unsafe { (*(*(*qc).dev).link).ap };
let data_addr: *mut () = unsafe { (*ap).ioaddr.data_addr };
let mut words: u32 = buflen >> 1i32;
// Transfer multiple of 2 bytes
if rw == 0 {
unsafe { ioread16_rep(data_addr as *const (), buf as *mut (), words as u64) };
} else {
unsafe { iowrite16_rep(data_addr, buf as *const (), words as u64) };
}
// Transfer trailing byte, if any.
if unlikely__3!(buflen & 1) != 0 {
let mut pad: [u8; 2] = {
let mut __t0: [u8; 2] = unsafe { core::mem::zeroed() };
__t0
};
// Point buf to the tail of buffer
let __v0: *mut u8 = buf;
buf = __v0.wrapping_offset(buflen.wrapping_sub(1) as isize);
// Use io*16_rep() accessors here as well to avoid pointlessly
// swapping bytes to and from on the big endian machines...
if rw == 0 {
unsafe { ioread16_rep(data_addr as *const (), pad.as_mut_ptr() as *mut (), 1u64) };
unsafe {
*buf = pad[0];
}
} else {
pad[0] = unsafe { *buf };
unsafe { iowrite16_rep(data_addr, pad.as_mut_ptr() as *const (), 1u64) };
}
words = words.wrapping_add(1);
}
words << 1i32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_data_xfer_861: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd, *mut u8, u32, i32) -> u32>, *mut ()>(Some(ata_sff_data_xfer)) };

// ata_sff_data_xfer32 - Transfer data by PIO
// @qc: queued command
// @buf: data buffer
// @buflen: buffer length
// @rw: read/write
//
// Transfer data from/to the device data register by PIO using 32bit
// I/O operations.
//
// LOCKING:
// Inherited from caller.
//
// RETURNS:
// Bytes consumed.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_data_xfer32")]
extern "C" fn ata_sff_data_xfer32(qc: *mut ata_queued_cmd, mut buf: *mut u8, buflen: u32, rw: i32) -> u32 {
let dev: *mut ata_device = unsafe { (*qc).dev };
let ap: *mut ata_port = unsafe { (*(*dev).link).ap };
let data_addr: *mut () = unsafe { (*ap).ioaddr.data_addr };
let words: u32 = buflen >> 2i32;
let slop: i32 = (buflen & 3) as i32;
if (unsafe { (*ap).pflags }) & 1048576 == 0 {
return ata_sff_data_xfer(qc, buf, buflen, rw);
}
// Transfer multiple of 4 bytes
if rw == 0 {
unsafe { ioread32_rep(data_addr as *const (), buf as *mut (), words as u64) };
} else {
unsafe { iowrite32_rep(data_addr, buf as *const (), words as u64) };
}
// Transfer trailing bytes, if any
if unlikely__2!(slop) != 0 {
let mut pad: [u8; 4] = {
let mut __t0: [u8; 4] = unsafe { core::mem::zeroed() };
__t0
};
// Point buf to the tail of buffer
let __v0: *mut u8 = buf;
buf = __v0.wrapping_offset(buflen.wrapping_sub(slop as u32) as isize);
// Use io*_rep() accessors here as well to avoid pointlessly
// swapping bytes to and from on the big endian machines...
if rw == 0 {
if slop < 3 {
unsafe { ioread16_rep(data_addr as *const (), pad.as_mut_ptr() as *mut (), 1u64) };
} else {
unsafe { ioread32_rep(data_addr as *const (), pad.as_mut_ptr() as *mut (), 1u64) };
}
unsafe { memcpy(buf as *mut (), pad.as_mut_ptr() as *const (), slop as u64) };
} else {
unsafe { memcpy(pad.as_mut_ptr() as *mut (), buf as *const (), slop as u64) };
if slop < 3 {
unsafe { iowrite16_rep(data_addr, pad.as_mut_ptr() as *const (), 1u64) };
} else {
unsafe { iowrite32_rep(data_addr, pad.as_mut_ptr() as *const (), 1u64) };
}
}
}
buflen.wrapping_add(1) & 4294967294u32
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_data_xfer32_862: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd, *mut u8, u32, i32) -> u32>, *mut ()>(Some(ata_sff_data_xfer32)) };

fn ata_pio_xfer(qc: *mut ata_queued_cmd, page: *mut page, offset: u32, xfer_size: u64) {
let do_write: bool = (unsafe { (*qc).tf.flags }) & 8 != 0;
let buf: *mut u8 = kmap_atomic(page as *const page) as *mut u8;
unsafe { unsafe { (*(*(*qc).ap).ops).sff_data_xfer }.unwrap()(qc, buf.wrapping_offset(offset as isize), xfer_size as u32, do_write as i32) };
loop {
loop {
if __same_type!() != 0 {
unsafe { __compiletime_assert_863() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__kunmap_atomic(buf as *const ());
if true {
break;
}
}
let __v2: bool = !do_write && PageSlab(page as *const page) == 0;
if __v2 {
flush_dcache_page(page);
}
}

// ata_pio_sector - Transfer a sector of data.
// @qc: Command on going
//
// Transfer qc->sect_size bytes of data from/to the ATA device.
//
// LOCKING:
// Inherited from caller.
fn ata_pio_sector(qc: *mut ata_queued_cmd) {
let ap: *mut ata_port = unsafe { (*qc).ap };
let mut page: *mut page = unsafe { core::mem::zeroed() };
let mut offset: u32 = 0;
let mut count: u32 = 0;
if unsafe { (*qc).cursg }.is_null() {
unsafe {
(*qc).curbytes = (*qc).nbytes;
}
return;
}
if (unsafe { (*qc).curbytes }) == unsafe { (*qc).nbytes }.wrapping_sub(unsafe { (*qc).sect_size }) {
unsafe {
(*ap).hsm_task_state = 3;
}
}
page = sg_page(unsafe { (*qc).cursg });
offset = unsafe { (*(*qc).cursg).offset }.wrapping_add(unsafe { (*qc).cursg_ofs });
// get the current page and offset
let __v0: *mut page = page;
page = __v0.wrapping_offset((offset >> 12i32) as isize);
let __v2: u32 = offset;
offset = ((__v2 as u64) % (_AC!() << 12i32)) as u32;
// don't overrun current sg
let mut __v4: u32;
let __UNIQUE_ID_x__864: u32 = unsafe { (*(*qc).cursg).length }.wrapping_sub(unsafe { (*qc).cursg_ofs });
let __UNIQUE_ID_y__865: u32 = unsafe { (*qc).sect_size };
while false {
}
__v4 = if __UNIQUE_ID_x__864 < __UNIQUE_ID_y__865 { __UNIQUE_ID_x__864 } else { __UNIQUE_ID_y__865 };
count = __v4;
trace_ata_sff_pio_transfer_data(qc, offset, count);
// Split the transfer when it splits a page boundary.  Note that the
// split still has to be dword aligned like all ATA data transfers.
let mut __v6: i64;
let __ret_warn_on: i32 = (offset % 4 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("867: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 867b - ., 3; .popsection\n# {0}", const 867 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_pio_sector_7, sym __slate_asm_string_ata_pio_sector_8, const 629 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("868: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 868b - ., 4; .popsection\n# {0}", const 868 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
if (offset.wrapping_add(count) as u64) > _AC!() << 12i32 {
let split_len: u32 = (_AC!() << 12i32).wrapping_sub(offset as u64) as u32;
ata_pio_xfer(qc, page, offset, split_len as u64);
ata_pio_xfer(qc, page.wrapping_offset(1 as isize), 0u32, count.wrapping_sub(split_len) as u64);
} else {
ata_pio_xfer(qc, page, offset, count as u64);
}
let __v12: u32 = unsafe { (*qc).curbytes };
unsafe {
(*qc).curbytes = __v12.wrapping_add(count);
}
let __v15: u32 = unsafe { (*qc).cursg_ofs };
unsafe {
(*qc).cursg_ofs = __v15.wrapping_add(count);
}
if (unsafe { (*qc).cursg_ofs }) == unsafe { (*(*qc).cursg).length } {
unsafe {
(*qc).cursg = sg_next((*qc).cursg);
}
if unsafe { (*qc).cursg }.is_null() {
unsafe {
(*ap).hsm_task_state = 3;
}
}
unsafe {
(*qc).cursg_ofs = 0;
}
}
}

// ata_pio_sectors - Transfer one or many sectors.
// @qc: Command on going
//
// Transfer one or many sectors of data from/to the
// ATA device for the DRQ request.
//
// LOCKING:
// Inherited from caller.
fn ata_pio_sectors(qc: *mut ata_queued_cmd) {
if is_multi_taskfile(unsafe { core::ptr::addr_of_mut!((*qc).tf) }) != 0 {
// READ/WRITE MULTIPLE
let mut nsect: u32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*(*qc).dev).multi_count }) == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("869: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 869b - ., 3; .popsection\n# {0}", const 869 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_pio_sectors_1, sym __slate_asm_string_ata_pio_sectors_2, const 666 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("870: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 870b - ., 4; .popsection\n# {0}", const 870 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
let mut __v5: u32;
let __UNIQUE_ID_x__871: u32 = unsafe { (*qc).nbytes }.wrapping_sub(unsafe { (*qc).curbytes }) / unsafe { (*qc).sect_size };
let __UNIQUE_ID_y__872: u32 = unsafe { (*(*qc).dev).multi_count };
while false {
}
__v5 = if __UNIQUE_ID_x__871 < __UNIQUE_ID_y__872 { __UNIQUE_ID_x__871 } else { __UNIQUE_ID_y__872 };
nsect = __v5;
loop {
let __v7: u32 = nsect;
nsect = __v7.wrapping_sub(1);
if __v7 == 0 {
break;
}
ata_pio_sector(qc);
}
} else {
ata_pio_sector(qc);
}
ata_sff_sync(unsafe { (*qc).ap }); // flush
}

/// atapi_send_cdb - Write CDB bytes to hardware
/// @ap: Port to which ATAPI device is attached.
/// @qc: Taskfile currently active
///
/// When device has indicated its readiness to accept
/// a CDB, this function is called.  Send the CDB.
///
/// LOCKING:
/// caller.
fn atapi_send_cdb(ap: *mut ata_port, qc: *mut ata_queued_cmd) {
// send SCSI cdb
trace_atapi_send_cdb(qc, 0u32, unsafe { (*(*qc).dev).cdb_len });
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*(*qc).dev).cdb_len }) < 12) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("874: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 874b - ., 3; .popsection\n# {0}", const 874 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_atapi_send_cdb_1, sym __slate_asm_string_atapi_send_cdb_2, const 693 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("875: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 875b - ., 4; .popsection\n# {0}", const 875 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { unsafe { (*(*ap).ops).sff_data_xfer }.unwrap()(qc, (*qc).cdb.as_mut_ptr() as *mut u8, (*(*qc).dev).cdb_len, 1i32) };
ata_sff_sync(ap);
// FIXME: If the CDB is for DMA do we need to do the transition delay
// or is bmdma_start guaranteed to do it ?
match (unsafe { (*qc).tf.protocol }) as i32 {
9 => {
unsafe {
(*ap).hsm_task_state = 2;
}
}
8 => {
unsafe {
(*ap).hsm_task_state = 3;
}
}
10 => {
unsafe {
(*ap).hsm_task_state = 3;
}
// initiate bmdma
trace_ata_bmdma_start(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe { unsafe { (*(*ap).ops).bmdma_start }.unwrap()(qc) };
}
_ => {
loop {
unsafe {
core::arch::asm!("876: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 876b - ., 3; .popsection\n# {0}", const 876 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_atapi_send_cdb_6, sym __slate_asm_string_atapi_send_cdb_7, const 715 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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

// __atapi_pio_bytes - Transfer data from/to the ATAPI device.
// @qc: Command on going
// @bytes: number of bytes
//
// Transfer data from/to the ATAPI device.
//
// LOCKING:
// Inherited from caller.
fn __atapi_pio_bytes(qc: *mut ata_queued_cmd, mut bytes: u32) -> i32 {
let mut __UNIQUE_ID_y__885: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__884: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__881: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__880: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__878: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__877: u32 = unsafe { core::mem::zeroed() };
let mut consumed: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(consumed) as *mut u32;
let mut count: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(count) as *mut u32;
let mut offset: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u32 = core::ptr::addr_of_mut!(offset) as *mut u32;
let mut buf: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut u8 = core::ptr::addr_of_mut!(buf) as *mut *mut u8;
let mut page: core::mem::MaybeUninit<*mut page> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut page = core::ptr::addr_of_mut!(page) as *mut *mut page;
let mut sg: core::mem::MaybeUninit<*mut scatterlist> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut *mut scatterlist = core::ptr::addr_of_mut!(sg) as *mut *mut scatterlist;
let mut ehi: *mut ata_eh_info = unsafe { core::mem::zeroed() };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
let mut rw: i32 = unsafe { core::mem::zeroed() };
unsafe {
rw = if (*qc).tf.flags & 8 != 0 { 1 } else { 0 };
ap = (*qc).ap;
dev = (*qc).dev;
ehi = core::ptr::addr_of_mut!((*(*dev).link).eh_info);
let __slate_auto_bytes = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
loop {
*__slate_slot_27 = (*qc).cursg;
if unlikely!((*__slate_slot_27).is_null()) != 0 {
ata_ehi_push_desc(ehi, c"unexpected or too much trailing data buf=%u cur=%u bytes=%u".as_ptr() as *const u8, (*qc).nbytes, (*qc).curbytes, bytes);
return 1i32.wrapping_neg();
} else {
*__slate_slot_26 = sg_page(*__slate_slot_27);
*__slate_slot_24 = unsafe { (*(*__slate_slot_27)).offset }.wrapping_add((*qc).cursg_ofs);
// get the current page and offset
let __v21: *mut page = *__slate_slot_26;
*__slate_slot_26 = __v21.wrapping_offset((*__slate_slot_24 >> 12i32) as isize);
let __v19: u32 = *__slate_slot_24;
let __v18: u32 = ((__v19 as u64) % (_AC!() << 12i32)) as u32;
*__slate_slot_24 = __v18;
__UNIQUE_ID_x__877 = unsafe { (*(*__slate_slot_27)).length }.wrapping_sub((*qc).cursg_ofs);
__UNIQUE_ID_y__878 = bytes;
while false {
}
// don't overrun current sg
*__slate_slot_23 = if __UNIQUE_ID_x__877 < __UNIQUE_ID_y__878 { __UNIQUE_ID_x__877 } else { __UNIQUE_ID_y__878 };
__UNIQUE_ID_x__880 = *__slate_slot_23;
__UNIQUE_ID_y__881 = ((_AC!() << 12i32) as u32).wrapping_sub(*__slate_slot_24);
while false {
}
// don't cross page boundaries
*__slate_slot_23 = if __UNIQUE_ID_x__880 < __UNIQUE_ID_y__881 { __UNIQUE_ID_x__880 } else { __UNIQUE_ID_y__881 };
trace_atapi_pio_transfer_data(qc, *__slate_slot_24, *__slate_slot_23);
// do the actual data transfer
*__slate_slot_25 = kmap_atomic(*__slate_slot_26 as *const page) as *mut u8;
*__slate_slot_22 = unsafe { (*(*ap).ops).sff_data_xfer }.unwrap()(qc, (*__slate_slot_25).wrapping_offset(*__slate_slot_24 as isize), *__slate_slot_23, rw);
loop {
if true {
__kunmap_atomic(*__slate_slot_25 as *const ());
if true {
break;
}
}
}
__UNIQUE_ID_x__884 = bytes;
__UNIQUE_ID_y__885 = *__slate_slot_22;
while false {
}
bytes = bytes.wrapping_sub(if __UNIQUE_ID_x__884 < __UNIQUE_ID_y__885 { __UNIQUE_ID_x__884 } else { __UNIQUE_ID_y__885 });
let __v4: u32 = (*qc).curbytes;
(*qc).curbytes = __v4.wrapping_add(*__slate_slot_23);
let __v1: u32 = (*qc).cursg_ofs;
(*qc).cursg_ofs = __v1.wrapping_add(*__slate_slot_23);
if (*qc).cursg_ofs == (*(*__slate_slot_27)).length {
(*qc).cursg = sg_next((*qc).cursg);
(*qc).cursg_ofs = 0;
}
// There used to be a  WARN_ON_ONCE(qc->cursg && count != consumed);
// Unfortunately __atapi_pio_bytes doesn't know enough to do the WARN
// check correctly as it doesn't know if it is the last request being
// made. Somebody should implement a proper sanity check.
if bytes == 0 {
break;
}
}
}
return 0;
}
}

/// atapi_pio_bytes - Transfer data from/to the ATAPI device.
/// @qc: Command on going
///
/// Transfer Transfer data from/to the ATAPI device.
///
/// LOCKING:
/// Inherited from caller.
fn atapi_pio_bytes(qc: *mut ata_queued_cmd) {
let mut do_write: i32 = unsafe { core::mem::zeroed() };
let mut i_write: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(i_write) as *mut i32;
let mut bytes: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(bytes) as *mut u32;
let mut bc_hi: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(bc_hi) as *mut u32;
let mut bc_lo: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(bc_lo) as *mut u32;
let mut ireason: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u32 = core::ptr::addr_of_mut!(ireason) as *mut u32;
let mut ehi: *mut ata_eh_info = unsafe { core::mem::zeroed() };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
ap = (*qc).ap;
dev = (*qc).dev;
ehi = core::ptr::addr_of_mut!((*(*dev).link).eh_info);
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
do_write = if (*qc).tf.flags & 8 != 0 { 1 } else { 0 };
// Abuse qc->result_tf for temp storage of intermediate TF
// here to save some kernel stack usage.
// For normal completion, qc->result_tf is not relevant. For
// error, qc->result_tf is later overwritten by ata_qc_complete().
// So, the correctness of qc->result_tf is not affected.
unsafe { (*(*ap).ops).sff_tf_read }.unwrap()(ap, core::ptr::addr_of_mut!((*qc).result_tf));
*__slate_slot_8 = (*qc).result_tf.nsect as u32;
*__slate_slot_7 = (*qc).result_tf.lbam as u32;
*__slate_slot_6 = (*qc).result_tf.lbah as u32;
*__slate_slot_5 = *__slate_slot_6 << 8i32 | *__slate_slot_7;
// shall be cleared to zero, indicating xfer of data
if unlikely__3!(*__slate_slot_8 & 1) == 0 {
// make sure transfer direction matches expected
*__slate_slot_4 = if *__slate_slot_8 & 2 == 0 { 1 } else { 0 };
if unlikely!(do_write != *__slate_slot_4) == 0 {
if unlikely!(*__slate_slot_5 == 0) == 0 {
if unlikely__2!(__atapi_pio_bytes(qc, *__slate_slot_5)) != 0 {
break '__join_0;
} else {
ata_sff_sync(ap); // flush
return;
}
}
}
}
ata_ehi_push_desc(ehi, c"ATAPI check failed (ireason=0x%x bytes=%u)".as_ptr() as *const u8, *__slate_slot_8, *__slate_slot_5);
}
(*qc).err_mask = (*qc).err_mask | 2;
(*ap).hsm_task_state = 4;
}
}

/// ata_hsm_ok_in_wq - Check if the qc can be handled in the workqueue.
/// @ap: the target ata_port
/// @qc: qc on going
///
/// RETURNS:
/// 1 if ok in workqueue, 0 otherwise.
fn ata_hsm_ok_in_wq(ap: *mut ata_port, qc: *mut ata_queued_cmd) -> i32 {
if (unsafe { (*qc).tf.flags }) & 64 != 0 {
return 1;
}
if (unsafe { (*ap).hsm_task_state }) == 1 {
if ((unsafe { (*qc).tf.protocol }) as i32) == 1 && (unsafe { (*qc).tf.flags }) & 8 != 0 {
return 1;
}
if ata_is_atapi(unsafe { (*qc).tf.protocol }) && (unsafe { (*(*qc).dev).flags }) & 4 == 0 {
return 1;
}
}
0
}

/// ata_hsm_qc_complete - finish a qc running on standard HSM
/// @qc: Command to complete
/// @in_wq: 1 if called from workqueue, 0 otherwise
///
/// Finish @qc which is running on standard HSM.
///
/// LOCKING:
/// If @in_wq is zero, spin_lock_irqsave(host lock).
/// Otherwise, none on entry and grabs host lock.
fn ata_hsm_qc_complete(mut qc: *mut ata_queued_cmd, in_wq: i32) {
let ap: *mut ata_port = unsafe { (*qc).ap };
if in_wq != 0 {
// EH might have kicked in while host lock is released.
qc = ata_qc_from_tag(ap, unsafe { (*qc).tag });
if !qc.is_null() {
if likely!((unsafe { (*qc).err_mask }) & 2 == 0) != 0 {
ata_sff_irq_on(ap);
unsafe { ata_qc_complete(qc) };
} else {
unsafe { ata_port_freeze(ap) };
}
}
} else {
if likely!((unsafe { (*qc).err_mask }) & 2 == 0) != 0 {
unsafe { ata_qc_complete(qc) };
} else {
unsafe { ata_port_freeze(ap) };
}
}
}

/// ata_sff_hsm_move - move the HSM to the next state.
/// @ap: the target ata_port
/// @qc: qc on going
/// @status: current device status
/// @in_wq: 1 if called from workqueue, 0 otherwise
///
/// RETURNS:
/// 1 when poll next status needed, 0 otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_hsm_move")]
extern "C" fn ata_sff_hsm_move(ap: *mut ata_port, qc: *mut ata_queued_cmd, mut status: u8, in_wq: i32) -> i32 {
let mut bug: core::mem::MaybeUninit<*mut bug_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut bug_entry = core::ptr::addr_of_mut!(bug) as *mut *mut bug_entry;
let mut __v2: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut bug_entry, ...)>> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)> = core::ptr::addr_of_mut!(__v2) as *mut Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
let mut poll_next: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_52: *mut i32 = core::ptr::addr_of_mut!(poll_next) as *mut i32;
let mut ehi: *mut ata_eh_info = unsafe { core::mem::zeroed() };
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
unsafe {
{
link = (*(*qc).dev).link;
ehi = core::ptr::addr_of_mut!((*link).eh_info);
let __slate_auto_bytes = __slate_slot_52 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
lockdep_assert_held__2!(unsafe { (*ap).lock });
let __ret_warn_on: i32 = ((*qc).flags & 1 == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("887: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 887b - ., 3; .popsection\n# {0}", const 887 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_sff_hsm_move_61, sym __slate_asm_string_ata_sff_hsm_move_62, const 924 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("888: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 888b - ., 4; .popsection\n# {0}", const 888 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
{
let __ret_warn_on: i32 = (in_wq != ata_hsm_ok_in_wq(ap, qc)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("889: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 889b - ., 3; .popsection\n# {0}", const 889 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_sff_hsm_move_59, sym __slate_asm_string_ata_sff_hsm_move_60, const 930 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("890: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 890b - ., 4; .popsection\n# {0}", const 890 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
}
// Make sure ata_sff_qc_issue() does not throw things
// like DMA polling into the workqueue. Notice that
// in_wq is not equivalent to (qc->tf.flags & ATA_TFLAG_POLLING).
'__join_0: {
'__join_40: {
'__join_12: {
'__join_13: {
'__join_9: {
'__join_4: {
loop {
trace_ata_sff_hsm_state(qc, status);
let __t1: u32 = (*ap).hsm_task_state;
if __t1 == 1 {
*__slate_slot_52 = ((*qc).tf.flags & 64) as i32; // Send first data block or PACKET CDB
// If polling, we will stay in the work queue after
// sending the data. Otherwise, interrupt handler
// takes over after sending the data.
// check device status
if unlikely!((status as i32) & 8 == 0) != 0 {
// handle BSY=0, DRQ=0 as error
if likely__3!((status as i32) & (1 | 32)) != 0 {
(*qc).err_mask = (*qc).err_mask | 1;
} else {
// HSM violation. Let EH handle this
ata_ehi_push_desc(ehi, c"ST_FIRST: !(DRQ|ERR|DF)".as_ptr() as *const u8);
(*qc).err_mask = (*qc).err_mask | 2;
}
// device stops HSM for abort/error
(*ap).hsm_task_state = 4;
} else {
// Device should not ask for data transfer (DRQ=1)
// when it finds something wrong.
// We ignore DRQ here and stop the HSM by
// changing hsm_task_state to HSM_ST_ERR and
// let the EH abort the command or reset the device.
if unlikely__2!((status as i32) & (1 | 32)) != 0 {
// Some ATAPI tape drives forget to clear the ERR bit
// when doing the next command (mostly request sense).
// We ignore ERR here to workaround and proceed sending
// the CDB.
if (*(*qc).dev).quirks & 128 == 0 {
ata_ehi_push_desc(ehi, c"ST_FIRST: DRQ=1 with device error, dev_stat 0x%X".as_ptr() as *const u8, status as i32);
(*qc).err_mask = (*qc).err_mask | 2;
(*ap).hsm_task_state = 4;
} else {
break '__join_40;
}
} else {
break '__join_40;
}
}
} else {
if __t1 == 2 {
if ((*qc).tf.protocol as i32) == 9 {
// ATAPI PIO protocol
if (status as i32) & 8 == 0 {
// No more data to transfer or device error.
// Device error will be tagged in HSM_ST_LAST.
(*ap).hsm_task_state = 3;
} else {
// Device should not ask for data transfer (DRQ=1)
// when it finds something wrong.
// We ignore DRQ here and stop the HSM by
// changing hsm_task_state to HSM_ST_ERR and
// let the EH abort the command or reset the device.
if unlikely__2!((status as i32) & (1 | 32)) != 0 {
ata_ehi_push_desc(ehi, c"ST-ATAPI: DRQ=1 with device error, dev_stat 0x%X".as_ptr() as *const u8, status as i32);
(*qc).err_mask = (*qc).err_mask | 2;
(*ap).hsm_task_state = 4;
} else {
atapi_pio_bytes(qc);
if unlikely!((*ap).hsm_task_state == 4) == 0 {
break '__join_13;
}
}
}
} else {
// ATA PIO protocol
if unlikely!((status as i32) & 8 == 0) != 0 {
// handle BSY=0, DRQ=0 as error
if likely__3!((status as i32) & (1 | 32)) != 0 {
// device stops HSM for abort/error
(*qc).err_mask = (*qc).err_mask | 1;
// If diagnostic failed and this is
// IDENTIFY, it's likely a phantom
// device.  Mark hint.
if (*(*qc).dev).quirks & 1 != 0 {
(*qc).err_mask = (*qc).err_mask | 512;
}
} else {
// HSM violation. Let EH handle this.
// Phantom devices also trigger this
// condition.  Mark hint.
ata_ehi_push_desc(ehi, c"ST-ATA: DRQ=0 without device error, dev_stat 0x%X".as_ptr() as *const u8, status as i32);
(*qc).err_mask = (*qc).err_mask | 514u32;
}
(*ap).hsm_task_state = 4;
} else {
// For PIO reads, some devices may ask for
// data transfer (DRQ=1) alone with ERR=1.
// We respect DRQ here and transfer one
// block of junk data before changing the
// hsm_task_state to HSM_ST_ERR.
//
// For PIO writes, ERR=1 DRQ=1 doesn't make
// sense since the data block has been
// transferred to the device.
if unlikely__2!((status as i32) & (1 | 32)) != 0 {
// data might be corrputed
(*qc).err_mask = (*qc).err_mask | 1;
if (*qc).tf.flags & 8 == 0 {
ata_pio_sectors(qc);
status = ata_wait_idle(ap);
}
if (status as i32) & (128 | 8) != 0 {
ata_ehi_push_desc(ehi, c"ST-ATA: BUSY|DRQ persists on ERR|DF, dev_stat 0x%X".as_ptr() as *const u8, status as i32);
(*qc).err_mask = (*qc).err_mask | 2;
}
// There are oddball controllers with
// status register stuck at 0x7f and
// lbal/m/h at zero which makes it
// pass all other presence detection
// mechanisms we have.  Set NODEV_HINT
// for it.  Kernel bz#7241.
if (status as i32) == 127 {
(*qc).err_mask = (*qc).err_mask | 512;
}
// ata_pio_sectors() might change the
// state to HSM_ST_LAST. so, the state
// is changed after ata_pio_sectors().
(*ap).hsm_task_state = 4;
} else {
ata_pio_sectors(qc);
if (*ap).hsm_task_state == 3 && (*qc).tf.flags & 8 == 0 {
status = ata_sff_busy_wait(ap, 136u32, 10u32);
if (status as i32) != 255 && (status as i32) & (128 | 8) != 0 {
(*qc).tf.flags = (*qc).tf.flags | 64;
ata_sff_queue_pio_task(link, 0u64);
return 0;
}
} else {
break '__join_12;
}
}
}
}
} else {
if __t1 == 3 {
if unlikely!(!ata_ok(status)) != 0 {
let mut __v12: u32 = __ac_err_mask(status);
(*qc).err_mask = (*qc).err_mask | __v12;
(*ap).hsm_task_state = 4;
} else {
break '__join_9;
}
} else {
if __t1 == 4 {
break '__join_4;
} else {
break;
}
}
}
}
}
*__slate_slot_52 = 0;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
__WARN_validate_printf(c"ata%d: SFF host state machine in invalid state %d".as_ptr() as *const u8, (*ap).print_id, (*ap).hsm_task_state);
*__slate_slot_2 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_ata_sff_hsm_move_55, sym __slate_asm_string_ata_sff_hsm_move_56, const 1163 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
*__slate_slot_0 = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
(*__slate_slot_2).unwrap()(*__slate_slot_0, (*ap).print_id, (*ap).hsm_task_state);
core::arch::asm!("", options(att_syntax));
if true {
break;
}
}
}
break '__join_0;
}
(*ap).hsm_task_state = 0;
// complete taskfile transaction
ata_hsm_qc_complete(qc, in_wq);
*__slate_slot_52 = 0;
break '__join_0;
}
// no more data to transfer
trace_ata_sff_hsm_command_complete(qc, status);
let __ret_warn_on: i32 = ((*qc).err_mask & 3u32 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("891: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 891b - ., 3; .popsection\n# {0}", const 891 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_sff_hsm_move_57, sym __slate_asm_string_ata_sff_hsm_move_58, const 1142 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("892: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 892b - ., 4; .popsection\n# {0}", const 892 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
(*ap).hsm_task_state = 0;
// complete taskfile transaction
ata_hsm_qc_complete(qc, in_wq);
*__slate_slot_52 = 0;
break '__join_0;
}
// bad ireason reported by device
}
// complete command or read/write the data register
*__slate_slot_52 = 1;
break '__join_0;
}
if ((*qc).tf.protocol as i32) == 1 {
// PIO data out protocol.
// send first data block.
// ata_pio_sectors() might change the state
// to HSM_ST_LAST. so, the state is changed here
// before ata_pio_sectors().
(*ap).hsm_task_state = 2;
ata_pio_sectors(qc);
} else {
atapi_send_cdb(ap, qc);
}
// send CDB
// if polling, ata_sff_pio_task() handles the rest.
// otherwise, interrupt handler takes over from here.
}
return *__slate_slot_52;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_893: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_hsm_move_894: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, *mut ata_queued_cmd, u8, i32) -> i32>, *mut ()>(Some(ata_sff_hsm_move)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_queue_work")]
extern "C" fn ata_sff_queue_work(work: *mut work_struct) {
queue_work(unsafe { ata_sff_wq }, work);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_queue_work_895: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut work_struct)>, *mut ()>(Some(ata_sff_queue_work)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_queue_delayed_work")]
extern "C" fn ata_sff_queue_delayed_work(dwork: *mut delayed_work, delay: u64) {
queue_delayed_work(unsafe { ata_sff_wq }, dwork, delay);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_queue_delayed_work_896: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut delayed_work, u64)>, *mut ()>(Some(ata_sff_queue_delayed_work)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_queue_pio_task")]
extern "C" fn ata_sff_queue_pio_task(link: *mut ata_link, delay: u64) {
let ap: *mut ata_port = unsafe { (*link).ap };
let mut __v0: i64;
let __ret_warn_on: i32 = (!unsafe { (*ap).sff_pio_task_link }.is_null() && (unsafe { (*ap).sff_pio_task_link }) != link) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("897: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 897b - ., 3; .popsection\n# {0}", const 897 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_sff_queue_pio_task_1, sym __slate_asm_string_ata_sff_queue_pio_task_2, const 1187 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("898: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 898b - ., 4; .popsection\n# {0}", const 898 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
(*ap).sff_pio_task_link = link;
}
// may fail if ata_sff_flush_pio_task() in progress
ata_sff_queue_delayed_work(unsafe { core::ptr::addr_of_mut!((*ap).sff_pio_task) }, msecs_to_jiffies(delay as u32));
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_queue_pio_task_899: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, u64)>, *mut ()>(Some(ata_sff_queue_pio_task)) };

#[unsafe(no_mangle)]
extern "C" fn ata_sff_flush_pio_task(ap: *mut ata_port) {
trace_ata_sff_flush_pio_task(ap);
unsafe { cancel_delayed_work_sync(core::ptr::addr_of_mut!((*ap).sff_pio_task)) };
// We wanna reset the HSM state to IDLE.  If we do so without
// grabbing the port lock, critical sections protected by it which
// expect the HSM state to stay stable may get surprised.  For
// example, we may set IDLE in between the time
// __ata_sff_port_intr() checks for HSM_ST_IDLE and before it calls
// ata_sff_hsm_move() causing ata_sff_hsm_move() to BUG().
spin_lock_irq(unsafe { (*ap).lock });
unsafe {
(*ap).hsm_task_state = 0;
}
spin_unlock_irq(unsafe { (*ap).lock });
unsafe {
(*ap).sff_pio_task_link = core::ptr::null_mut::<ata_link>();
}
}

#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_pio_task")]
extern "C" fn ata_sff_pio_task(work: *mut work_struct) {
let mut poll_next: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(poll_next) as *mut i32;
let mut wait_mask: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u8 = core::ptr::addr_of_mut!(wait_mask) as *mut u8;
let mut status: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u8 = core::ptr::addr_of_mut!(status) as *mut u8;
let mut qc: core::mem::MaybeUninit<*mut ata_queued_cmd> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut ata_queued_cmd = core::ptr::addr_of_mut!(qc) as *mut *mut ata_queued_cmd;
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
unsafe {
'__slate_dispatch: {
ap = ((work as *const u8).wrapping_offset(-(offsetof__90!(ata_port) as isize)) as *mut ()) as *mut ata_port;
link = (*ap).sff_pio_task_link;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
spin_lock_irq((*ap).lock);
'__join_15: {
loop {
if unlikely!(unsafe { (*ap).sff_pio_task_link }.is_null()) != 0 {
break '__join_15;
} else {
if true {
break;
}
}
}
'__join_0: {
// qc can be NULL if timeout occurred
*__slate_slot_8 = ata_qc_from_tag(ap, (*link).active_tag);
if (*__slate_slot_8).is_null() {
(*ap).sff_pio_task_link = core::ptr::null_mut::<ata_link>();
} else {
loop {
let __ret_warn_on: i32 = ((*ap).hsm_task_state == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("901: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 901b - ., 3; .popsection\n# {0}", const 901 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_sff_pio_task_12, sym __slate_asm_string_ata_sff_pio_task_13, const 1236 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("902: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 902b - ., 4; .popsection\n# {0}", const 902 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_6 = 128;
if (*ap).hsm_task_state == 3 {
*__slate_slot_6 = ((*__slate_slot_6 as i32) | 8) as u8;
}
// This is purely heuristic.  This is a fast path.
// Sometimes when we enter, BSY will be cleared in
// a chk-status or two.  If not, the drive is probably seeking
// or something.  Snooze for a couple msecs, then
// chk-status again.  If still busy, queue delayed work.
*__slate_slot_7 = ata_sff_busy_wait(ap, *__slate_slot_6 as u32, 5u32);
if (*__slate_slot_7 as i32) & (*__slate_slot_6 as i32) != 0 {
spin_unlock_irq((*ap).lock);
ata_msleep(ap, 2u32);
spin_lock_irq((*ap).lock);
*__slate_slot_7 = ata_sff_busy_wait(ap, *__slate_slot_6 as u32, 10u32);
if (*__slate_slot_7 as i32) & (*__slate_slot_6 as i32) != 0 {
break;
}
}
// hsm_move() may trigger another command to be processed.
// clean the link beforehand.
(*ap).sff_pio_task_link = core::ptr::null_mut::<ata_link>();
// move the HSM
*__slate_slot_5 = ata_sff_hsm_move(ap, *__slate_slot_8, *__slate_slot_7, 1i32);
// another command or interrupt handler
// may be running at this point.
if *__slate_slot_5 == 0 {
break '__join_0;
}
}
ata_sff_queue_pio_task(link, 16u64);
}
}
spin_unlock_irq((*ap).lock);
break '__slate_dispatch;
}
core::arch::asm!("900: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 900b - ., 3; .popsection\n# {0}", const 900 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_ata_sff_pio_task_14, sym __slate_asm_string_ata_sff_pio_task_15, const 1227 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}

/// ata_sff_qc_issue - issue taskfile to a SFF controller
/// @qc: command to issue to device
///
/// This function issues a PIO or NODATA command to a SFF
/// controller.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// Zero on success, AC_ERR_* mask on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_qc_issue")]
extern "C" fn ata_sff_qc_issue(qc: *mut ata_queued_cmd) -> u32 {
let ap: *mut ata_port = unsafe { (*qc).ap };
let link: *mut ata_link = unsafe { (*(*qc).dev).link };
// Use polling pio if the LLD doesn't handle
// interrupt driven pio and atapi CDB interrupt.
if (unsafe { (*ap).flags }) & 512 != 0 {
unsafe {
(*qc).tf.flags = (*qc).tf.flags | 64;
}
}
// select the device
ata_dev_select(ap, unsafe { (*(*qc).dev).devno }, 1u32, 0u32);
// start the command
match (unsafe { (*qc).tf.protocol }) as i32 {
0 => {
if (unsafe { (*qc).tf.flags }) & 64 != 0 {
ata_qc_set_polling(qc);
}
ata_tf_to_host(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe {
(*ap).hsm_task_state = 3;
}
if (unsafe { (*qc).tf.flags }) & 64 != 0 {
ata_sff_queue_pio_task(link, 0u64);
}
}
1 => {
if (unsafe { (*qc).tf.flags }) & 64 != 0 {
ata_qc_set_polling(qc);
}
ata_tf_to_host(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
if (unsafe { (*qc).tf.flags }) & 8 != 0 {
// PIO data out protocol
unsafe {
(*ap).hsm_task_state = 1;
}
ata_sff_queue_pio_task(link, 0u64);
// always send first data block using the
// ata_sff_pio_task() codepath.
} else {
// PIO data in protocol
unsafe {
(*ap).hsm_task_state = 2;
}
if (unsafe { (*qc).tf.flags }) & 64 != 0 {
ata_sff_queue_pio_task(link, 0u64);
}
// if polling, ata_sff_pio_task() handles the
// rest.  otherwise, interrupt handler takes
// over from here.
}
}
8 | 9 => {
if (unsafe { (*qc).tf.flags }) & 64 != 0 {
ata_qc_set_polling(qc);
}
ata_tf_to_host(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe {
(*ap).hsm_task_state = 1;
}
// send cdb by polling if no cdb interrupt
if (unsafe { (*(*qc).dev).flags }) & 4 == 0 || (unsafe { (*qc).tf.flags }) & 64 != 0 {
ata_sff_queue_pio_task(link, 0u64);
}
}
_ => {
return 64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_qc_issue_903: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>, *mut ()>(Some(ata_sff_qc_issue)) };

/// ata_sff_qc_fill_rtf - fill result TF using ->sff_tf_read
/// @qc: qc to fill result TF for
///
/// @qc is finished and result TF needs to be filled.  Fill it
/// using ->sff_tf_read.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_qc_fill_rtf")]
extern "C" fn ata_sff_qc_fill_rtf(qc: *mut ata_queued_cmd) {
unsafe { unsafe { (*(*(*qc).ap).ops).sff_tf_read }.unwrap()((*qc).ap, core::ptr::addr_of_mut!((*qc).result_tf)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_qc_fill_rtf_904: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd)>, *mut ()>(Some(ata_sff_qc_fill_rtf)) };

fn ata_sff_idle_irq(__v0: *mut ata_port) -> u32 {
let __v1: u64 = unsafe { (*__v0).stats.idle_irq };
unsafe {
(*__v0).stats.idle_irq = __v1.wrapping_add(1);
}
return 0; // irq not handled
}

fn __ata_sff_port_intr(ap: *mut ata_port, qc: *mut ata_queued_cmd, hsmv_on_idle: bool) -> u32 {
let mut status: u8 = 0;
trace_ata_sff_port_intr(qc, hsmv_on_idle as u8);
// Check whether we are expecting interrupt in this state
match unsafe { (*ap).hsm_task_state } {
1 => {
if (unsafe { (*(*qc).dev).flags }) & 4 == 0 {
return ata_sff_idle_irq(ap);
}
// Some pre-ATAPI-4 devices assert INTRQ
// at this state when ready to receive CDB.
// Check the ATA_DFLAG_CDB_INTR flag is enough here.
// The flag was turned on only for atapi devices.  No
// need to check ata_is_atapi(qc->tf.protocol) again.
}
0 => {
return ata_sff_idle_irq(ap);
}
_ => {
}
}
// check main status, clearing INTRQ if needed
status = ata_sff_irq_status(ap);
if (status as i32) & 128 != 0 {
if hsmv_on_idle {
// BMDMA engine is already stopped, we're screwed
unsafe {
(*qc).err_mask = (*qc).err_mask | 2;
}
unsafe {
(*ap).hsm_task_state = 4;
}
} else {
return ata_sff_idle_irq(ap);
}
}
// clear irq events
if (unsafe { (*(*ap).ops).sff_irq_clear }) != None {
unsafe { unsafe { (*(*ap).ops).sff_irq_clear }.unwrap()(ap) };
}
ata_sff_hsm_move(ap, qc, status, 0i32);
return 1; // irq handled
}

/// ata_sff_port_intr - Handle SFF port interrupt
/// @ap: Port on which interrupt arrived (possibly...)
/// @qc: Taskfile currently active in engine
///
/// Handle port interrupt for given queued command.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// One if interrupt was handled, zero if not (shared irq).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_port_intr")]
extern "C" fn ata_sff_port_intr(ap: *mut ata_port, qc: *mut ata_queued_cmd) -> u32 {
__ata_sff_port_intr(ap, qc, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_port_intr_905: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, *mut ata_queued_cmd) -> u32>, *mut ()>(Some(ata_sff_port_intr)) };

fn __ata_sff_interrupt(irq: i32, dev_instance: *mut (), port_intr: Option<unsafe extern "C" fn(*mut ata_port, *mut ata_queued_cmd) -> u32>) -> u32 {
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
let mut retry: bool = unsafe { core::mem::zeroed() };
let mut qc: core::mem::MaybeUninit<*mut ata_queued_cmd> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut *mut ata_queued_cmd = core::ptr::addr_of_mut!(qc) as *mut *mut ata_queued_cmd;
let mut ap_1: *mut ata_port = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut polling: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut u32 = core::ptr::addr_of_mut!(polling) as *mut u32;
let mut idle: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut u32 = core::ptr::addr_of_mut!(idle) as *mut u32;
let mut handled: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut u32 = core::ptr::addr_of_mut!(handled) as *mut u32;
let mut i: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u32 = core::ptr::addr_of_mut!(i) as *mut u32;
let mut retried: bool = unsafe { core::mem::zeroed() };
let mut host: *mut ata_host = unsafe { core::mem::zeroed() };
unsafe {
host = dev_instance as *mut ata_host;
retried = false;
let __slate_auto_bytes_1 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
// TODO: make _irqsave conditional on x86 PCI IDE legacy mode
spin_lock_irqsave!(core::ptr::addr_of_mut!((*host).lock), *__slate_slot_18);
loop {
*__slate_slot_19 = 0;
*__slate_slot_20 = 0;
*__slate_slot_21 = 0;
*__slate_slot_22 = 0;
while *__slate_slot_22 < (*host).n_ports {
ap_1 = *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(*__slate_slot_22 as isize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_16 = ata_qc_from_tag(ap_1, (*ap_1).link.active_tag);
if !(*__slate_slot_16).is_null() {
if (*(*__slate_slot_16)).tf.flags & 64 == 0 {
let __v11: u32 = port_intr.unwrap()(ap_1, *__slate_slot_16);
*__slate_slot_21 |= __v11;
} else {
*__slate_slot_19 |= (1i32 << *__slate_slot_22) as u32;
}
} else {
*__slate_slot_20 |= (1i32 << *__slate_slot_22) as u32;
}
let __v8: u32 = *__slate_slot_22;
*__slate_slot_22 = __v8.wrapping_add(1);
}
// If no port was expecting IRQ but the controller is actually
// asserting IRQ line, nobody cared will ensue.  Check IRQ
// pending status if available and clear spurious IRQ.
if *__slate_slot_21 == 0 && !retried {
retry = false;
*__slate_slot_22 = 0;
while *__slate_slot_22 < (*host).n_ports {
ap = *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(*__slate_slot_22 as isize);
if *__slate_slot_19 & ((1i32 << *__slate_slot_22) as u32) == 0 {
if (*(*ap).ops).sff_irq_check == None {
__v4 = true;
} else {
__v4 = !unsafe { (*(*ap).ops).sff_irq_check }.unwrap()(ap);
}
if !__v4 {
if *__slate_slot_20 & ((1i32 << *__slate_slot_22) as u32) != 0 {
unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap);
if (*(*ap).ops).sff_irq_clear != None {
unsafe { (*(*ap).ops).sff_irq_clear }.unwrap()(ap);
}
} else {
// clear INTRQ and check if BUSY cleared
if (unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap) as i32) & 128 == 0 {
retry = (retry as i32) | 1 != 0;
}
// With command in flight, we can't do
// sff_irq_clear() w/o racing with completion.
}
}
}
let __v1: u32 = *__slate_slot_22;
*__slate_slot_22 = __v1.wrapping_add(1);
}
if retry {
retried = true;
} else {
break;
}
} else {
break;
}
}
spin_unlock_irqrestore(core::ptr::addr_of_mut!((*host).lock), *__slate_slot_18);
return IRQ_RETVAL!(*__slate_slot_21 != 0) as u32;
}
}

/// ata_sff_interrupt - Default SFF ATA host interrupt handler
/// @irq: irq line (unused)
/// @dev_instance: pointer to our ata_host information structure
///
/// Default interrupt handler for PCI IDE devices.  Calls
/// ata_sff_port_intr() for each port that is not disabled.
///
/// LOCKING:
/// Obtains host lock during operation.
///
/// RETURNS:
/// IRQ_NONE or IRQ_HANDLED.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_interrupt")]
extern "C" fn ata_sff_interrupt(irq: i32, dev_instance: *mut ()) -> u32 {
__ata_sff_interrupt(irq, dev_instance, Some(ata_sff_port_intr))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_interrupt_906: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, *mut ()>(Some(ata_sff_interrupt)) };

// ata_sff_lost_interrupt	-	Check for an apparent lost interrupt
// @ap: port that appears to have timed out
//
// Called from the libata error handlers when the core code suspects
// an interrupt has been lost. If it has complete anything we can and
// then return. Interface must support altstatus for this faster
// recovery to occur.
//
// Locking:
// Caller holds host lock
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_lost_interrupt")]
extern "C" fn ata_sff_lost_interrupt(ap: *mut ata_port) {
let mut status: u8 = 0;
// Only one outstanding command per SFF channel
let qc: *mut ata_queued_cmd = ata_qc_from_tag(ap, unsafe { (*ap).link.active_tag });
// We cannot lose an interrupt on a non-existent or polled command
if qc.is_null() || (unsafe { (*qc).tf.flags }) & 64 != 0 {
return;
}
// See if the controller thinks it is still busy - if so the command
// isn't a lost IRQ but is still in progress
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!ata_sff_altstatus(ap, core::ptr::addr_of_mut!(status)))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("907: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 907b - ., 3; .popsection\n# {0}", const 907 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_sff_lost_interrupt_1, sym __slate_asm_string_ata_sff_lost_interrupt_2, const 1588 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("908: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 908b - ., 4; .popsection\n# {0}", const 908 as i32, options(nostack, att_syntax));
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
if (status as i32) & 128 != 0 {
return;
}
// There was a command running, we are no longer busy and we have
// no interrupt.
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x014ata%u: lost interrupt (Status 0x%x)\n".as_ptr() as *const u8, (*ap).print_id, status as i32) };
// Run the host interrupt logic as if the interrupt had not been
// lost
ata_sff_port_intr(ap, qc);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_lost_interrupt_909: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_sff_lost_interrupt)) };

/// ata_sff_freeze - Freeze SFF controller port
/// @ap: port to freeze
///
/// Freeze SFF controller port.
///
/// LOCKING:
/// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_freeze")]
extern "C" fn ata_sff_freeze(ap: *mut ata_port) {
unsafe {
(*ap).ctl = (((*ap).ctl as i32) | 2) as u8;
}
unsafe {
(*ap).last_ctl = (*ap).ctl;
}
ata_sff_set_devctl(ap, unsafe { (*ap).ctl });
// Under certain circumstances, some controllers raise IRQ on
// ATA_NIEN manipulation.  Also, many controllers fail to mask
// previously pending IRQ on ATA_NIEN assertion.  Clear it.
unsafe { unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap) };
if (unsafe { (*(*ap).ops).sff_irq_clear }) != None {
unsafe { unsafe { (*(*ap).ops).sff_irq_clear }.unwrap()(ap) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_freeze_910: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_sff_freeze)) };

/// ata_sff_thaw - Thaw SFF controller port
/// @ap: port to thaw
///
/// Thaw SFF controller port.
///
/// LOCKING:
/// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_thaw")]
extern "C" fn ata_sff_thaw(ap: *mut ata_port) {
// clear & re-enable interrupts
unsafe { unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap) };
if (unsafe { (*(*ap).ops).sff_irq_clear }) != None {
unsafe { unsafe { (*(*ap).ops).sff_irq_clear }.unwrap()(ap) };
}
ata_sff_irq_on(ap);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_thaw_911: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_sff_thaw)) };

/// ata_sff_prereset - prepare SFF link for reset
/// @link: SFF link to be reset
/// @deadline: deadline jiffies for the operation
///
/// SFF link @link is about to be reset.  Initialize it.  It first
/// calls ata_std_prereset() and wait for !BSY if the port is
/// being softreset.
///
/// LOCKING:
/// Kernel thread context (may sleep)
///
/// RETURNS:
/// Always 0.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_prereset")]
extern "C" fn ata_sff_prereset(link: *mut ata_link, deadline: u64) -> i32 {
let ehc: *mut ata_eh_context = unsafe { core::ptr::addr_of_mut!((*link).eh_context) };
let mut rc: i32 = 0;
// The standard prereset is best-effort and always returns 0
unsafe { ata_std_prereset(link, deadline) };
// if we're about to do hardreset, nothing more to do
if (unsafe { (*ehc).i.action }) & 4 != 0 {
return 0;
}
// wait for !BSY if we don't know that no device is attached
if !unsafe { ata_link_offline(link) } {
rc = ata_sff_wait_ready(link, deadline);
if rc != 0 && rc != 19i32.wrapping_neg() {
loop {
if sata_pmp_attached(unsafe { (*link).ap }) || !unsafe { (*(*link).ap).slave_link }.is_null() {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014ata%u.%02u: device not ready (errno=%d), forcing hardreset\n".as_ptr() as *const u8, (*(*link).ap).print_id, (*link).pmp, rc) };
} else {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014ata%u: device not ready (errno=%d), forcing hardreset\n".as_ptr() as *const u8, (*(*link).ap).print_id, rc) };
}
if true {
break;
}
}
unsafe {
(*ehc).i.action = (*ehc).i.action | 4;
}
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_prereset_912: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, u64) -> i32>, *mut ()>(Some(ata_sff_prereset)) };

/// ata_devchk - PATA device presence detection
/// @ap: ATA channel to examine
/// @device: Device to examine (starting at zero)
///
/// This technique was originally described in
/// Hale Landis's ATADRVR (www.ata-atapi.com), and
/// later found its way into the ATA/ATAPI spec.
///
/// Write a pattern to the ATA shadow registers,
/// and if a device is present, it will respond by
/// correctly storing and echoing back the
/// ATA shadow register contents.
///
/// RETURN:
/// true if device is present, false if not.
///
/// LOCKING:
/// caller.
fn ata_devchk(ap: *mut ata_port, device: u32) -> bool {
let ioaddr: *mut ata_ioports = unsafe { core::ptr::addr_of_mut!((*ap).ioaddr) };
let mut nsect: u8 = 0;
let mut lbal: u8 = 0;
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, device) };
unsafe { iowrite8(85u8, (*ioaddr).nsect_addr) };
unsafe { iowrite8(170u8, (*ioaddr).lbal_addr) };
unsafe { iowrite8(170u8, (*ioaddr).nsect_addr) };
unsafe { iowrite8(85u8, (*ioaddr).lbal_addr) };
unsafe { iowrite8(85u8, (*ioaddr).nsect_addr) };
unsafe { iowrite8(170u8, (*ioaddr).lbal_addr) };
nsect = (unsafe { ioread8((*ioaddr).nsect_addr as *const ()) }) as u8;
lbal = (unsafe { ioread8((*ioaddr).lbal_addr as *const ()) }) as u8;
if (nsect as i32) == 85 && (lbal as i32) == 170 {
return true;
}
// we found a device
return false; // nothing found
}

/// ata_sff_dev_classify - Parse returned ATA device signature
/// @dev: ATA device to classify (starting at zero)
/// @present: device seems present
/// @r_err: Value of error register on completion
///
/// After an event -- SRST, E.D.D., or SATA COMRESET -- occurs,
/// an ATA/ATAPI-defined set of values is placed in the ATA
/// shadow registers, indicating the results of device detection
/// and diagnostics.
///
/// Select the ATA device, and read the values from the ATA shadow
/// registers.  Then parse according to the Error register value,
/// and the spec-defined values examined by ata_dev_classify().
///
/// LOCKING:
/// caller.
///
/// RETURNS:
/// Device type - %ATA_DEV_ATA, %ATA_DEV_ATAPI or %ATA_DEV_NONE.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_dev_classify")]
extern "C" fn ata_sff_dev_classify(dev: *mut ata_device, present: i32, r_err: *mut u8) -> u32 {
let ap: *mut ata_port = unsafe { (*(*dev).link).ap };
let mut tf: ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tf) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
}
let mut class: u32 = 0;
let mut err: u8 = 0;
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, (*dev).devno) };
unsafe { memset(core::ptr::addr_of_mut!(tf) as *mut (), 0i32, core::mem::size_of::<ata_taskfile>() as u64) };
unsafe { unsafe { (*(*ap).ops).sff_tf_read }.unwrap()(ap, core::ptr::addr_of_mut!(tf)) };
err = unsafe { tf.__slate_anon_8.error };
if !r_err.is_null() {
unsafe {
*r_err = err;
}
}
// see if device passed diags: continue and warn later
if (err as i32) == 0 {
unsafe {
(*dev).quirks = (*dev).quirks | 1;
}
} else {
if (err as i32) != 1 {
if !((unsafe { (*dev).devno }) == 0 && (err as i32) == 129) {
return 11;
}
}
}
// diagnostic fail : do nothing _YET_
// do nothing
// do nothing
// determine if device is ATA or ATAPI
class = unsafe { ata_port_classify(ap, core::ptr::addr_of_mut!(tf) as *const ata_taskfile) };
match class {
0 => {
if present != 0 && (unsafe { (*dev).quirks }) & 1 != 0 {
class = 1;
} else {
class = 11;
}
// If the device failed diagnostic, it's likely to
// have reported incorrect device signature too.
// Assume ATA device if the device seems present but
// device signature is invalid with diagnostic
// failure.
}
1 => {
if ((unsafe { unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap) }) as i32) == 0 {
class = 11;
}
}
_ => {
}
}
class
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_dev_classify_913: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_device, i32, *mut u8) -> u32>, *mut ()>(Some(ata_sff_dev_classify)) };

/// ata_sff_wait_after_reset - wait for devices to become ready after reset
/// @link: SFF link which is just reset
/// @devmask: mask of present devices
/// @deadline: deadline jiffies for the operation
///
/// Wait devices attached to SFF @link to become ready after
/// reset.  It contains preceding 150ms wait to avoid accessing TF
/// status register too early.
///
/// LOCKING:
/// Kernel thread context (may sleep).
///
/// RETURNS:
/// 0 on success, -ENODEV if some or all of devices in @devmask
/// don't seem to exist.  -errno on other errors.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_wait_after_reset")]
extern "C" fn ata_sff_wait_after_reset(link: *mut ata_link, devmask: u32, deadline: u64) -> i32 {
let ap: *mut ata_port = unsafe { (*link).ap };
let ioaddr: *mut ata_ioports = unsafe { core::ptr::addr_of_mut!((*ap).ioaddr) };
let dev0: u32 = devmask & 1u32;
let dev1: u32 = devmask & 2u32;
let mut rc: i32 = 0;
let mut ret: i32 = 0;
unsafe { ata_msleep(ap, 150u32) };
// always check readiness of the master device
rc = ata_sff_wait_ready(link, deadline);
// -ENODEV means the odd clown forgot the D7 pulldown resistor
// and TF status is 0xff, bail out on it too.
if rc != 0 {
return rc;
}
// if device 1 was found in ata_devchk, wait for register
// access briefly, then wait for BSY to clear.
if dev1 != 0 {
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, 1u32) };
// Wait for register access.  Some ATAPI devices fail
// to set nsect/lbal after reset, so don't waste too
// much time on it.  We're gonna wait for !BSY anyway.
for i in 0i32..2i32 {
let nsect: u8 = (unsafe { ioread8((*ioaddr).nsect_addr as *const ()) }) as u8;
let lbal: u8 = (unsafe { ioread8((*ioaddr).lbal_addr as *const ()) }) as u8;
if (nsect as i32) == 1 && (lbal as i32) == 1 {
break;
}
unsafe { ata_msleep(ap, 50u32) }; // give drive a breather
}
rc = ata_sff_wait_ready(link, deadline);
if rc != 0 {
if rc != 19i32.wrapping_neg() {
return rc;
}
ret = rc;
}
}
// is all this really necessary?
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, 0u32) };
if dev1 != 0 {
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, 1u32) };
}
if dev0 != 0 {
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, 0u32) };
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_wait_after_reset_914: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, u32, u64) -> i32>, *mut ()>(Some(ata_sff_wait_after_reset)) };

fn ata_bus_softreset(ap: *mut ata_port, devmask: u32, deadline: u64) -> i32 {
let ioaddr: *mut ata_ioports = unsafe { core::ptr::addr_of_mut!((*ap).ioaddr) };
if !unsafe { (*ap).ioaddr.ctl_addr }.is_null() {
// software reset.  causes dev0 to be selected
unsafe { iowrite8((*ap).ctl, (*ioaddr).ctl_addr) };
udelay(20u64); // FIXME: flush
unsafe { iowrite8((((*ap).ctl as i32) | 4) as u8, (*ioaddr).ctl_addr) };
udelay(20u64); // FIXME: flush
unsafe { iowrite8((*ap).ctl, (*ioaddr).ctl_addr) };
unsafe {
(*ap).last_ctl = (*ap).ctl;
}
}
// wait the port to become ready
ata_sff_wait_after_reset(unsafe { core::ptr::addr_of_mut!((*ap).link) }, devmask, deadline)
}

/// ata_sff_softreset - reset host port via ATA SRST
/// @link: ATA link to reset
/// @classes: resulting classes of attached devices
/// @deadline: deadline jiffies for the operation
///
/// Reset host port using ATA SRST.
///
/// LOCKING:
/// Kernel thread context (may sleep)
///
/// RETURNS:
/// 0 on success, -errno otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_softreset")]
extern "C" fn ata_sff_softreset(link: *mut ata_link, classes: *mut u32, deadline: u64) -> i32 {
let ap: *mut ata_port = unsafe { (*link).ap };
let slave_possible: u32 = ((unsafe { (*ap).flags }) & 1) as u32;
let mut devmask: u32 = 0;
let mut rc: i32 = 0;
let mut err: u8 = 0;
// determine if device 0/1 are present
if ata_devchk(ap, 0u32) {
devmask |= 1u32;
}
let __v2: bool = slave_possible != 0 && ata_devchk(ap, 1u32);
if __v2 {
devmask |= 2u32;
}
// select device 0 again
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, 0u32) };
// issue bus reset
rc = ata_bus_softreset(ap, devmask, deadline);
// if link is occupied, -ENODEV too is an error
let __v5: bool = rc != 0 && (rc != 19i32.wrapping_neg() || (unsafe { sata_scr_valid(link) }) != 0);
if __v5 {
loop {
if sata_pmp_attached(unsafe { (*link).ap }) || !unsafe { (*(*link).ap).slave_link }.is_null() {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x013ata%u.%02u: SRST failed (errno=%d)\n".as_ptr() as *const u8, (*(*link).ap).print_id, (*link).pmp, rc) };
} else {
let mut __v9: i32;
while false {
}
__v9 = unsafe { _printk(c"\x013ata%u: SRST failed (errno=%d)\n".as_ptr() as *const u8, (*(*link).ap).print_id, rc) };
}
if true {
break;
}
}
return rc;
}
// determine by signature whether we have ATA or ATAPI devices
unsafe {
*classes.wrapping_offset(0 as isize) = ata_sff_dev_classify(((*link).device.as_mut_ptr() as *mut ata_device).wrapping_offset(0 as isize), (devmask & 1u32) as i32, core::ptr::addr_of_mut!(err));
}
if slave_possible != 0 && (err as i32) != 129 {
unsafe {
*classes.wrapping_offset(1 as isize) = ata_sff_dev_classify(((*link).device.as_mut_ptr() as *mut ata_device).wrapping_offset(1 as isize), (devmask & 2u32) as i32, core::ptr::addr_of_mut!(err));
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_softreset_915: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, *mut u32, u64) -> i32>, *mut ()>(Some(ata_sff_softreset)) };

/// sata_sff_hardreset - reset host port via SATA phy reset
/// @link: link to reset
/// @class: resulting class of attached device
/// @deadline: deadline jiffies for the operation
///
/// SATA phy-reset host port using DET bits of SControl register,
/// wait for !BSY and classify the attached device.
///
/// LOCKING:
/// Kernel thread context (may sleep)
///
/// RETURNS:
/// 0 on success, -errno otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.sata_sff_hardreset")]
extern "C" fn sata_sff_hardreset(link: *mut ata_link, class: *mut u32, deadline: u64) -> i32 {
let ehc: *mut ata_eh_context = unsafe { core::ptr::addr_of_mut!((*link).eh_context) };
let timing: *const u32 = sata_ehc_deb_timing(ehc);
let mut online: bool = false;
let rc: i32 = unsafe { sata_link_hardreset(link, timing, deadline, core::ptr::addr_of_mut!(online), Some(ata_sff_check_ready)) };
if online {
unsafe {
*class = ata_sff_dev_classify((*link).device.as_mut_ptr() as *mut ata_device, 1i32, core::ptr::null_mut::<u8>());
}
}
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_sata_sff_hardreset_916: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, *mut u32, u64) -> i32>, *mut ()>(Some(sata_sff_hardreset)) };

/// ata_sff_postreset - SFF postreset callback
/// @link: the target SFF ata_link
/// @classes: classes of attached devices
///
/// This function is invoked after a successful reset.  It first
/// calls ata_std_postreset() and performs SFF specific postreset
/// processing.
///
/// LOCKING:
/// Kernel thread context (may sleep)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_postreset")]
extern "C" fn ata_sff_postreset(link: *mut ata_link, classes: *mut u32) {
let ap: *mut ata_port = unsafe { (*link).ap };
unsafe { ata_std_postreset(link, classes) };
// is double-select really necessary?
if (unsafe { *classes.wrapping_offset(0 as isize) }) != 11 {
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, 1u32) };
}
if (unsafe { *classes.wrapping_offset(1 as isize) }) != 11 {
unsafe { unsafe { (*(*ap).ops).sff_dev_select }.unwrap()(ap, 0u32) };
}
// bail out if no device is present
if (unsafe { *classes.wrapping_offset(0 as isize) }) == 11 && (unsafe { *classes.wrapping_offset(1 as isize) }) == 11 {
return;
}
// set up device control
if ata_sff_set_devctl(ap, unsafe { (*ap).ctl }) {
unsafe {
(*ap).last_ctl = (*ap).ctl;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_postreset_917: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_link, *mut u32)>, *mut ()>(Some(ata_sff_postreset)) };

// ata_sff_drain_fifo - Stock FIFO drain logic for SFF controllers
// @qc: command
//
// Drain the FIFO and device of any stuck data following a command
// failing to complete. In some cases this is necessary before a
// reset will recover the device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_drain_fifo")]
extern "C" fn ata_sff_drain_fifo(qc: *mut ata_queued_cmd) {
// Drain up to 64K of data before we give up this recovery method
let mut count: i32 = 0;
// We only need to flush incoming data when a command was running
if qc.is_null() || (unsafe { (*qc).dma_dir }) == 1 {
return;
}
let ap: *mut ata_port = unsafe { (*qc).ap };
while ((unsafe { unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap) }) as i32) & 8 != 0 && count < 65536 {
unsafe { ioread16((*ap).ioaddr.data_addr as *const ()) };
count = count.wrapping_add(2);
}
if count != 0 {
let mut __v3: i32;
if false {
unsafe { _printk(c"\x017ata%u: drained %d bytes to clear DRQ\n".as_ptr() as *const u8, (*ap).print_id, count) };
}
__v3 = 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_drain_fifo_918: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd)>, *mut ()>(Some(ata_sff_drain_fifo)) };

/// ata_sff_error_handler - Stock error handler for SFF controller
/// @ap: port to handle error for
///
/// Stock error handler for SFF controller.  It can handle both
/// PATA and SATA controllers.  Many controllers should be able to
/// use this EH as-is or with some added handling before and
/// after.
///
/// LOCKING:
/// Kernel thread context (may sleep)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_error_handler")]
extern "C" fn ata_sff_error_handler(ap: *mut ata_port) {
let mut qc: *mut ata_queued_cmd = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
qc = __ata_qc_from_tag(ap, unsafe { (*ap).link.active_tag });
if !qc.is_null() && (unsafe { (*qc).flags }) & 65536 == 0 {
qc = core::ptr::null_mut::<ata_queued_cmd>();
}
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
// We *MUST* do FIFO draining before we issue a reset as
// several devices helpfully clear their internal state and
// will lock solid if we touch the data port post reset. Pass
// qc in case anyone wants to do different PIO/DMA recovery or
// has per command fixups
if (unsafe { (*(*ap).ops).sff_drain_fifo }) != None {
unsafe { unsafe { (*(*ap).ops).sff_drain_fifo }.unwrap()(qc) };
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
unsafe { ata_std_error_handler(ap) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_error_handler_919: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_sff_error_handler)) };

/// ata_sff_std_ports - initialize ioaddr with standard port offsets.
/// @ioaddr: IO address structure to be initialized
///
/// Utility function which initializes data_addr, error_addr,
/// feature_addr, nsect_addr, lbal_addr, lbam_addr, lbah_addr,
/// device_addr, status_addr, and command_addr to standard offsets
/// relative to cmd_addr.
///
/// Does not set ctl_addr, altstatus_addr, bmdma_addr, or scr_addr.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_sff_std_ports")]
extern "C" fn ata_sff_std_ports(ioaddr: *mut ata_ioports) {
unsafe {
(*ioaddr).data_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(0 as isize) as *mut ();
}
unsafe {
(*ioaddr).error_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(1 as isize) as *mut ();
}
unsafe {
(*ioaddr).feature_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(1 as isize) as *mut ();
}
unsafe {
(*ioaddr).nsect_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(2 as isize) as *mut ();
}
unsafe {
(*ioaddr).lbal_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(3 as isize) as *mut ();
}
unsafe {
(*ioaddr).lbam_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(4 as isize) as *mut ();
}
unsafe {
(*ioaddr).lbah_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(5 as isize) as *mut ();
}
unsafe {
(*ioaddr).device_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(6 as isize) as *mut ();
}
unsafe {
(*ioaddr).status_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(7 as isize) as *mut ();
}
unsafe {
(*ioaddr).command_addr = ((*ioaddr).cmd_addr as *const u8).wrapping_offset(7 as isize) as *mut ();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sff_std_ports_920: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_ioports)>, *mut ()>(Some(ata_sff_std_ports)) };

fn ata_resources_present(pdev: *mut pci_dev, mut port: i32) -> bool {
// Check the PCI resources for this channel are enabled
port = port.wrapping_mul(2);
for i in 0i32..2i32 {
let __v2: bool = pci_resource_start!(pdev, port.wrapping_add(i)) == 0 || pci_resource_len!(pdev, port.wrapping_add(i)) == 0;
if __v2 {
return false;
}
}
true
}

/// ata_pci_sff_init_host - acquire native PCI ATA resources and init host
/// @host: target ATA host
///
/// Acquire native PCI ATA resources for @host and initialize the
/// first two ports of @host accordingly.  Ports marked dummy are
/// skipped and allocation failure makes the port dummy.
///
/// Note that native PCI resources are valid even for legacy hosts
/// as we fix up pdev resources array early in boot, so this
/// function can be used for both native and legacy SFF hosts.
///
/// LOCKING:
/// Inherited from calling layer (may sleep).
///
/// RETURNS:
/// 0 if at least one port is initialized, -ENODEV if no port is
/// available.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_pci_sff_init_host")]
extern "C" fn ata_pci_sff_init_host(host: *mut ata_host) -> i32 {
let gdev: *mut device = unsafe { (*host).dev };
let pdev: *mut pci_dev = to_pci_dev!(gdev);
let mut mask: u32 = 0;
let mut rc: i32 = 0;
// request, iomap BARs and init port addresses accordingly
for i in 0i32..2i32 {
let ap: *mut ata_port = unsafe { *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize) };
let base: i32 = i.wrapping_mul(2);
if ata_port_is_dummy(ap) == 0 {
// Discard disabled ports.  Some controllers show
// their unused channels this way.  Disabled ports are
// made dummy.
if !ata_resources_present(pdev, i) {
unsafe {
(*ap).ops = core::ptr::addr_of_mut!(ata_dummy_port_ops);
}
} else {
rc = unsafe { pcim_iomap_regions(pdev, 3i32 << base, dev_driver_string(gdev as *const device)) };
if rc != 0 {
while false {
}
unsafe { _dev_warn(gdev as *const device, c"failed to request/iomap BARs for port %d (errno=%d)\n".as_ptr() as *const u8, i, rc) };
if rc == 16i32.wrapping_neg() {
unsafe { pcim_pin_device(pdev) };
}
unsafe {
(*ap).ops = core::ptr::addr_of_mut!(ata_dummy_port_ops);
}
} else {
let __v3: *const *mut () = unsafe { pcim_iomap_table(pdev) };
let iomap: *const *mut () = __v3;
unsafe {
(*host).iomap = __v3;
}
unsafe {
(*ap).ioaddr.cmd_addr = *iomap.wrapping_offset(base as isize);
}
let __v4: *mut () = (((unsafe { *iomap.wrapping_offset(base.wrapping_add(1) as isize) }) as u64) | 2) as *mut ();
unsafe {
(*ap).ioaddr.ctl_addr = __v4;
}
unsafe {
(*ap).ioaddr.altstatus_addr = __v4;
}
ata_sff_std_ports(unsafe { core::ptr::addr_of_mut!((*ap).ioaddr) });
unsafe { ata_port_desc(ap, c"cmd 0x%llx ctl 0x%llx".as_ptr() as *const u8, pci_resource_start!(pdev, base), pci_resource_start!(pdev, base.wrapping_add(1))) };
mask |= (1i32 << i) as u32;
}
}
}
}
if mask == 0 {
while false {
}
unsafe { _dev_err(gdev as *const device, c"no available native port\n".as_ptr() as *const u8) };
return 19i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_pci_sff_init_host_921: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host) -> i32>, *mut ()>(Some(ata_pci_sff_init_host)) };

/// ata_pci_sff_prepare_host - helper to prepare PCI PIO-only SFF ATA host
/// @pdev: target PCI device
/// @ppi: array of port_info, must be enough for two ports
/// @r_host: out argument for the initialized ATA host
///
/// Helper to allocate PIO-only SFF ATA host for @pdev, acquire
/// all PCI resources and initialize it accordingly in one go.
///
/// LOCKING:
/// Inherited from calling layer (may sleep).
///
/// RETURNS:
/// 0 on success, -errno otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_pci_sff_prepare_host")]
extern "C" fn ata_pci_sff_prepare_host(pdev: *mut pci_dev, ppi: *const *const ata_port_info, r_host: *mut *mut ata_host) -> i32 {
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut host: core::mem::MaybeUninit<*mut ata_host> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut ata_host = core::ptr::addr_of_mut!(host) as *mut *mut ata_host;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if devres_open_group(core::ptr::addr_of_mut!((*pdev).dev), core::ptr::null_mut::<()>(), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)).is_null() {
return 12i32.wrapping_neg();
} else {
*__slate_slot_1 = ata_host_alloc_pinfo(core::ptr::addr_of_mut!((*pdev).dev), ppi, 2i32);
if (*__slate_slot_1).is_null() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"failed to allocate ATA host\n".as_ptr() as *const u8);
*__slate_slot_0 = 12i32.wrapping_neg();
} else {
*__slate_slot_0 = ata_pci_sff_init_host(*__slate_slot_1);
if *__slate_slot_0 == 0 {
devres_remove_group(core::ptr::addr_of_mut!((*pdev).dev), core::ptr::null_mut::<()>());
*r_host = *__slate_slot_1;
return 0;
}
}
devres_release_group(core::ptr::addr_of_mut!((*pdev).dev), core::ptr::null_mut::<()>());
return *__slate_slot_0;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_pci_sff_prepare_host_922: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *const *const ata_port_info, *mut *mut ata_host) -> i32>, *mut ()>(Some(ata_pci_sff_prepare_host)) };

/// ata_pci_sff_activate_host - start SFF host, request IRQ and register it
/// @host: target SFF ATA host
/// @irq_handler: irq_handler used when requesting IRQ(s)
/// @sht: scsi_host_template to use when registering the host
///
/// This is the counterpart of ata_host_activate() for SFF ATA
/// hosts.  This separate helper is necessary because SFF hosts
/// use two separate interrupts in legacy mode.
///
/// LOCKING:
/// Inherited from calling layer (may sleep).
///
/// RETURNS:
/// 0 on success, -errno otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_pci_sff_activate_host")]
extern "C" fn ata_pci_sff_activate_host(host: *mut ata_host, irq_handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, sht: *const scsi_host_template) -> i32 {
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut mask: u8 = unsafe { core::mem::zeroed() };
let mut tmp8: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u8 = core::ptr::addr_of_mut!(tmp8) as *mut u8;
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut legacy_mode: i32 = unsafe { core::mem::zeroed() };
let mut drv_name: *const u8 = unsafe { core::mem::zeroed() };
let mut pdev: *mut pci_dev = unsafe { core::mem::zeroed() };
let mut dev: *mut device = unsafe { core::mem::zeroed() };
unsafe {
dev = (*host).dev;
pdev = to_pci_dev!(dev);
drv_name = dev_driver_string((*host).dev as *const device);
legacy_mode = 0;
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
*__slate_slot_9 = ata_host_start(host);
if *__slate_slot_9 != 0 {
return *__slate_slot_9;
} else {
if (*pdev).class >> 8i32 == 257 {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
mask = 0;
// ATA spec says we should use legacy mode when one
// port is in legacy mode, but disabled ports on some
// PCI hosts appear as fixed legacy ports, e.g SB600/700
// on which the secondary port is not wired, so
// ignore ports that are marked as 'dummy' during
// this check
pci_read_config_byte(pdev as *const pci_dev, 9i32, core::ptr::addr_of_mut!(*__slate_slot_8));
if ata_port_is_dummy(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(0 as isize)) == 0 {
mask = ((mask as i32) | 1i32) as u8;
}
if ata_port_is_dummy(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(1 as isize)) == 0 {
mask = ((mask as i32) | 4i32) as u8;
}
if (*__slate_slot_8 as i32) & (mask as i32) != (mask as i32) {
legacy_mode = 1;
}
}
if devres_open_group(dev, core::ptr::null_mut::<()>(), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)).is_null() {
return 12i32.wrapping_neg();
} else {
'__join_3: {
if legacy_mode == 0 && (*pdev).irq != 0 {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_9 = devm_request_irq(dev, (*pdev).irq, irq_handler, 128u64, drv_name, host as *mut ());
if *__slate_slot_9 != 0 {
break '__join_3;
} else {
*__slate_slot_2 = 0;
while *__slate_slot_2 < 2 {
if ata_port_is_dummy(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(*__slate_slot_2 as isize)) == 0 {
ata_port_desc_misc(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(*__slate_slot_2 as isize), (*pdev).irq as i32);
}
let __v1: i32 = *__slate_slot_2;
*__slate_slot_2 = __v1.wrapping_add(1);
}
}
} else {
if legacy_mode != 0 {
if ata_port_is_dummy(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(0 as isize)) == 0 {
*__slate_slot_9 = devm_request_irq(dev, ATA_PRIMARY_IRQ!() as u32, irq_handler, 128u64, drv_name, host as *mut ());
if *__slate_slot_9 != 0 {
break '__join_3;
} else {
ata_port_desc_misc(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(0 as isize), ATA_PRIMARY_IRQ!());
}
}
if ata_port_is_dummy(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(1 as isize)) == 0 {
*__slate_slot_9 = devm_request_irq(dev, ATA_SECONDARY_IRQ!() as u32, irq_handler, 128u64, drv_name, host as *mut ());
if *__slate_slot_9 != 0 {
break '__join_3;
} else {
ata_port_desc_misc(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(1 as isize), ATA_SECONDARY_IRQ!());
}
}
}
}
*__slate_slot_9 = ata_host_register(host, sht);
}
if *__slate_slot_9 == 0 {
devres_remove_group(dev, core::ptr::null_mut::<()>());
} else {
devres_release_group(dev, core::ptr::null_mut::<()>());
}
return *__slate_slot_9;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_pci_sff_activate_host_923: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host, Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, *const scsi_host_template) -> i32>, *mut ()>(Some(ata_pci_sff_activate_host)) };

fn ata_sff_find_valid_pi(ppi: *const *const ata_port_info) -> *const ata_port_info {
// look up the first valid port_info
let mut i: i32 = 0;
while i < 2 && !unsafe { *ppi.wrapping_offset(i as isize) }.is_null() {
if (unsafe { (*(*ppi.wrapping_offset(i as isize))).port_ops }) != unsafe { core::ptr::addr_of_mut!(ata_dummy_port_ops) } {
return unsafe { *ppi.wrapping_offset(i as isize) };
}
i = i.wrapping_add(1);
}
core::ptr::null::<ata_port_info>()
}

fn ata_pci_init_one(pdev: *mut pci_dev, ppi: *const *const ata_port_info, sht: *const scsi_host_template, host_priv: *mut (), hflags: i32, bmdma: bool) -> i32 {
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut host: *mut ata_host = unsafe { core::mem::zeroed() };
let mut pi: core::mem::MaybeUninit<*const ata_port_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *const ata_port_info = core::ptr::addr_of_mut!(pi) as *mut *const ata_port_info;
let mut dev: *mut device = unsafe { core::mem::zeroed() };
unsafe {
dev = core::ptr::addr_of_mut!((*pdev).dev);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
host = core::ptr::null_mut::<ata_host>();
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_5 = ata_sff_find_valid_pi(ppi);
if (*__slate_slot_5).is_null() {
while false {
}
_dev_err(core::ptr::addr_of_mut!((*pdev).dev) as *const device, c"no valid port_info specified\n".as_ptr() as *const u8);
return 22i32.wrapping_neg();
} else {
if devres_open_group(dev, core::ptr::null_mut::<()>(), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)).is_null() {
return 12i32.wrapping_neg();
} else {
*__slate_slot_3 = pcim_enable_device(pdev);
if *__slate_slot_3 == 0 {
if bmdma {
*__slate_slot_3 = ata_pci_bmdma_prepare_host(pdev, ppi, core::ptr::addr_of_mut!(host));
} else {
*__slate_slot_3 = ata_pci_sff_prepare_host(pdev, ppi, core::ptr::addr_of_mut!(host));
}
// prepare and activate BMDMA host
// prepare and activate SFF host
if *__slate_slot_3 == 0 {
(*host).private_data = host_priv;
(*host).flags = (*host).flags | (hflags as u64);
if bmdma {
pci_set_master(pdev);
*__slate_slot_3 = ata_pci_sff_activate_host(host, Some(ata_bmdma_interrupt), sht);
} else {
*__slate_slot_3 = ata_pci_sff_activate_host(host, Some(ata_sff_interrupt), sht);
}
}
}
if *__slate_slot_3 == 0 {
devres_remove_group(core::ptr::addr_of_mut!((*pdev).dev), core::ptr::null_mut::<()>());
} else {
devres_release_group(core::ptr::addr_of_mut!((*pdev).dev), core::ptr::null_mut::<()>());
}
return *__slate_slot_3;
}
}
}
}

/// ata_pci_sff_init_one - Initialize/register PIO-only PCI IDE controller
/// @pdev: Controller to be initialized
/// @ppi: array of port_info, must be enough for two ports
/// @sht: scsi_host_template to use when registering the host
/// @host_priv: host private_data
/// @hflag: host flags
///
/// This is a helper function which can be called from a driver's
/// xxx_init_one() probe function if the hardware uses traditional
/// IDE taskfile registers and is PIO only.
///
/// ASSUMPTION:
/// Nobody makes a single channel controller that appears solely as
/// the secondary legacy port on PCI.
///
/// LOCKING:
/// Inherited from PCI layer (may sleep).
///
/// RETURNS:
/// Zero on success, negative on errno-based value on error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_pci_sff_init_one")]
extern "C" fn ata_pci_sff_init_one(pdev: *mut pci_dev, ppi: *const *const ata_port_info, sht: *const scsi_host_template, host_priv: *mut (), hflag: i32) -> i32 {
ata_pci_init_one(pdev, ppi, sht, host_priv, hflag, false)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_pci_sff_init_one_924: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *const *const ata_port_info, *const scsi_host_template, *mut (), i32) -> i32>, *mut ()>(Some(ata_pci_sff_init_one)) };

// BMDMA support
#[unsafe(no_mangle)]
static ata_bmdma_port_ops: __SlateConst<ata_port_operations> = __SlateConst({
let mut __t0: ata_port_operations = unsafe { core::mem::zeroed() };
__t0.qc_prep = Some(ata_bmdma_qc_prep);
__t0.qc_issue = Some(ata_bmdma_qc_issue);
__t0.error_handler = Some(ata_bmdma_error_handler);
__t0.post_internal_cmd = Some(ata_bmdma_post_internal_cmd);
__t0.port_start = Some(ata_bmdma_port_start);
__t0.sff_irq_clear = Some(ata_bmdma_irq_clear);
__t0.bmdma_setup = Some(ata_bmdma_setup);
__t0.bmdma_start = Some(ata_bmdma_start);
__t0.bmdma_stop = Some(ata_bmdma_stop);
__t0.bmdma_status = Some(ata_bmdma_status);
__t0.inherits = unsafe { core::ptr::addr_of!(ata_sff_port_ops.0) };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_port_ops_925: *mut () = (unsafe { core::ptr::addr_of!(ata_bmdma_port_ops.0) }) as *mut ();

#[unsafe(no_mangle)]
static ata_bmdma32_port_ops: __SlateConst<ata_port_operations> = __SlateConst({
let mut __t0: ata_port_operations = unsafe { core::mem::zeroed() };
__t0.port_start = Some(ata_bmdma_port_start32);
__t0.sff_data_xfer = Some(ata_sff_data_xfer32);
__t0.inherits = unsafe { core::ptr::addr_of!(ata_bmdma_port_ops.0) };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma32_port_ops_926: *mut () = (unsafe { core::ptr::addr_of!(ata_bmdma32_port_ops.0) }) as *mut ();

/// ata_bmdma_fill_sg - Fill PCI IDE PRD table
/// @qc: Metadata associated with taskfile to be transferred
///
/// Fill PCI IDE PRD (scatter-gather) table with segments
/// associated with the current disk command.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_bmdma_fill_sg(qc: *mut ata_queued_cmd) {
let ap: *mut ata_port = unsafe { (*qc).ap };
let prd: *mut ata_bmdma_prd = unsafe { (*ap).bmdma_prd };
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
let mut si: u32 = 0;
let mut pi: u32 = 0;
sg = unsafe { (*qc).sg };
while si < unsafe { (*qc).n_elem } {
let mut addr: u32 = 0;
let mut offset: u32 = 0;
let mut sg_len: u32 = 0;
let mut len: u32 = 0;
// determine if physical DMA addr spans 64K boundary.
// Note h/w doesn't support 64-bit, so we unconditionally
// truncate dma_addr_t to u32.
addr = sg_dma_address!(sg) as u32;
sg_len = sg_dma_len!(sg);
while sg_len != 0 {
offset = addr & 65535;
len = sg_len;
if offset.wrapping_add(sg_len) > 65536 {
len = 65536u32.wrapping_sub(offset);
}
unsafe {
(*prd.wrapping_offset(pi as isize)).addr = addr;
}
unsafe {
(*prd.wrapping_offset(pi as isize)).flags_len = len & 65535;
}
pi = pi.wrapping_add(1);
sg_len = sg_len.wrapping_sub(len);
addr = addr.wrapping_add(len);
}
si = si.wrapping_add(1);
sg = sg_next(sg);
}
let __v12: *mut ata_bmdma_prd = prd.wrapping_offset(pi.wrapping_sub(1) as isize);
unsafe {
(*__v12).flags_len = (*__v12).flags_len | 2147483648u32;
}
}

/// ata_bmdma_fill_sg_dumb - Fill PCI IDE PRD table
/// @qc: Metadata associated with taskfile to be transferred
///
/// Fill PCI IDE PRD (scatter-gather) table with segments
/// associated with the current disk command. Perform the fill
/// so that we avoid writing any length 64K records for
/// controllers that don't follow the spec.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_bmdma_fill_sg_dumb(qc: *mut ata_queued_cmd) {
let ap: *mut ata_port = unsafe { (*qc).ap };
let prd: *mut ata_bmdma_prd = unsafe { (*ap).bmdma_prd };
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
let mut si: u32 = 0;
let mut pi: u32 = 0;
sg = unsafe { (*qc).sg };
while si < unsafe { (*qc).n_elem } {
let mut addr: u32 = 0;
let mut offset: u32 = 0;
let mut sg_len: u32 = 0;
let mut len: u32 = 0;
let mut blen: u32 = 0;
// determine if physical DMA addr spans 64K boundary.
// Note h/w doesn't support 64-bit, so we unconditionally
// truncate dma_addr_t to u32.
addr = sg_dma_address!(sg) as u32;
sg_len = sg_dma_len!(sg);
while sg_len != 0 {
offset = addr & 65535;
len = sg_len;
if offset.wrapping_add(sg_len) > 65536 {
len = 65536u32.wrapping_sub(offset);
}
blen = len & 65535;
unsafe {
(*prd.wrapping_offset(pi as isize)).addr = addr;
}
if blen == 0 {
// Some PATA chipsets like the CS5530 can't
// cope with 0x0000 meaning 64K as the spec
// says
unsafe {
(*prd.wrapping_offset(pi as isize)).flags_len = 32768;
}
blen = 32768;
let __v2: u32 = pi.wrapping_add(1);
pi = __v2;
unsafe {
(*prd.wrapping_offset(__v2 as isize)).addr = addr.wrapping_add(32768);
}
}
unsafe {
(*prd.wrapping_offset(pi as isize)).flags_len = blen;
}
pi = pi.wrapping_add(1);
sg_len = sg_len.wrapping_sub(len);
addr = addr.wrapping_add(len);
}
si = si.wrapping_add(1);
sg = sg_next(sg);
}
let __v14: *mut ata_bmdma_prd = prd.wrapping_offset(pi.wrapping_sub(1) as isize);
unsafe {
(*__v14).flags_len = (*__v14).flags_len | 2147483648u32;
}
}

/// ata_bmdma_qc_prep - Prepare taskfile for submission
/// @qc: Metadata associated with taskfile to be prepared
///
/// Prepare ATA taskfile for submission.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_qc_prep")]
extern "C" fn ata_bmdma_qc_prep(qc: *mut ata_queued_cmd) -> u32 {
if (unsafe { (*qc).flags }) & 2 == 0 {
return 0;
}
ata_bmdma_fill_sg(qc);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_qc_prep_927: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>, *mut ()>(Some(ata_bmdma_qc_prep)) };

/// ata_bmdma_dumb_qc_prep - Prepare taskfile for submission
/// @qc: Metadata associated with taskfile to be prepared
///
/// Prepare ATA taskfile for submission.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_dumb_qc_prep")]
extern "C" fn ata_bmdma_dumb_qc_prep(qc: *mut ata_queued_cmd) -> u32 {
if (unsafe { (*qc).flags }) & 2 == 0 {
return 0;
}
ata_bmdma_fill_sg_dumb(qc);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_dumb_qc_prep_928: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>, *mut ()>(Some(ata_bmdma_dumb_qc_prep)) };

/// ata_bmdma_qc_issue - issue taskfile to a BMDMA controller
/// @qc: command to issue to device
///
/// This function issues a PIO, NODATA or DMA command to a
/// SFF/BMDMA controller.  PIO and NODATA are handled by
/// ata_sff_qc_issue().
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// Zero on success, AC_ERR_* mask on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_qc_issue")]
extern "C" fn ata_bmdma_qc_issue(qc: *mut ata_queued_cmd) -> u32 {
let ap: *mut ata_port = unsafe { (*qc).ap };
let link: *mut ata_link = unsafe { (*(*qc).dev).link };
// defer PIO handling to sff_qc_issue
if !ata_is_dma(unsafe { (*qc).tf.protocol }) {
return ata_sff_qc_issue(qc);
}
// select the device
ata_dev_select(ap, unsafe { (*(*qc).dev).devno }, 1u32, 0u32);
// start the command
match (unsafe { (*qc).tf.protocol }) as i32 {
2 => {
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*qc).tf.flags }) & 64 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("929: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 929b - ., 3; .popsection\n# {0}", const 929 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_bmdma_qc_issue_1, sym __slate_asm_string_ata_bmdma_qc_issue_2, const 2666 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("930: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 930b - ., 4; .popsection\n# {0}", const 930 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
trace_ata_tf_load(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile);
unsafe { unsafe { (*(*ap).ops).sff_tf_load }.unwrap()(ap, core::ptr::addr_of_mut!((*qc).tf) as *const ata_taskfile) }; // load tf registers
trace_ata_bmdma_setup(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe { unsafe { (*(*ap).ops).bmdma_setup }.unwrap()(qc) }; // set up bmdma
trace_ata_bmdma_start(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe { unsafe { (*(*ap).ops).bmdma_start }.unwrap()(qc) }; // initiate bmdma
unsafe {
(*ap).hsm_task_state = 3;
}
}
10 => {
let mut __v6: i64;
let __ret_warn_on: i32 = ((unsafe { (*qc).tf.flags }) & 64 != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("931: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 931b - ., 3; .popsection\n# {0}", const 931 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_bmdma_qc_issue_7, sym __slate_asm_string_ata_bmdma_qc_issue_8, const 2678 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("932: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 932b - ., 4; .popsection\n# {0}", const 932 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
trace_ata_tf_load(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile);
unsafe { unsafe { (*(*ap).ops).sff_tf_load }.unwrap()(ap, core::ptr::addr_of_mut!((*qc).tf) as *const ata_taskfile) }; // load tf registers
trace_ata_bmdma_setup(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe { unsafe { (*(*ap).ops).bmdma_setup }.unwrap()(qc) }; // set up bmdma
unsafe {
(*ap).hsm_task_state = 1;
}
// send cdb by polling if no cdb interrupt
if (unsafe { (*(*qc).dev).flags }) & 4 == 0 {
ata_sff_queue_pio_task(link, 0u64);
}
}
_ => {
let mut __v11: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("933: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 933b - ., 3; .popsection\n# {0}", const 933 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_bmdma_qc_issue_12, sym __slate_asm_string_ata_bmdma_qc_issue_13, const 2692 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("934: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 934b - ., 4; .popsection\n# {0}", const 934 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v11 = (__ret_warn_on != 0) as i64;
return 64;
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_qc_issue_935: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>, *mut ()>(Some(ata_bmdma_qc_issue)) };

/// ata_bmdma_port_intr - Handle BMDMA port interrupt
/// @ap: Port on which interrupt arrived (possibly...)
/// @qc: Taskfile currently active in engine
///
/// Handle port interrupt for given queued command.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// One if interrupt was handled, zero if not (shared irq).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_port_intr")]
extern "C" fn ata_bmdma_port_intr(ap: *mut ata_port, qc: *mut ata_queued_cmd) -> u32 {
let ehi: *mut ata_eh_info = unsafe { core::ptr::addr_of_mut!((*ap).link.eh_info) };
let mut host_stat: u8 = 0;
let mut bmdma_stopped: bool = false;
let mut handled: u32 = 0;
let __v0: bool = (unsafe { (*ap).hsm_task_state }) == 3 && ata_is_dma(unsafe { (*qc).tf.protocol });
if __v0 {
// check status of DMA engine
host_stat = unsafe { unsafe { (*(*ap).ops).bmdma_status }.unwrap()(ap) };
trace_ata_bmdma_status(ap, host_stat as u32);
// if it's not our irq...
if (host_stat as i32) & 4 == 0 {
return ata_sff_idle_irq(ap);
}
// before we do anything else, clear DMA-Start bit
trace_ata_bmdma_stop(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe { unsafe { (*(*ap).ops).bmdma_stop }.unwrap()(qc) };
bmdma_stopped = true;
if unlikely__2!((host_stat as i32) & 2) != 0 {
// error when transferring data to/from memory
unsafe {
(*qc).err_mask = (*qc).err_mask | 32;
}
unsafe {
(*ap).hsm_task_state = 4;
}
}
}
handled = __ata_sff_port_intr(ap, qc, bmdma_stopped);
let __v4: bool = unlikely__3!(unsafe { (*qc).err_mask }) != 0 && ata_is_dma(unsafe { (*qc).tf.protocol });
if __v4 {
unsafe { ata_ehi_push_desc(ehi, c"BMDMA stat 0x%x".as_ptr() as *const u8, host_stat as i32) };
}
handled
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_port_intr_936: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, *mut ata_queued_cmd) -> u32>, *mut ()>(Some(ata_bmdma_port_intr)) };

/// ata_bmdma_interrupt - Default BMDMA ATA host interrupt handler
/// @irq: irq line (unused)
/// @dev_instance: pointer to our ata_host information structure
///
/// Default interrupt handler for PCI IDE devices.  Calls
/// ata_bmdma_port_intr() for each port that is not disabled.
///
/// LOCKING:
/// Obtains host lock during operation.
///
/// RETURNS:
/// IRQ_NONE or IRQ_HANDLED.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_interrupt")]
extern "C" fn ata_bmdma_interrupt(irq: i32, dev_instance: *mut ()) -> u32 {
__ata_sff_interrupt(irq, dev_instance, Some(ata_bmdma_port_intr))
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_interrupt_937: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut ()) -> u32>, *mut ()>(Some(ata_bmdma_interrupt)) };

/// ata_bmdma_error_handler - Stock error handler for BMDMA controller
/// @ap: port to handle error for
///
/// Stock error handler for BMDMA controller.  It can handle both
/// PATA and SATA controllers.  Most BMDMA controllers should be
/// able to use this EH as-is or with some added handling before
/// and after.
///
/// LOCKING:
/// Kernel thread context (may sleep)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_error_handler")]
extern "C" fn ata_bmdma_error_handler(ap: *mut ata_port) {
let mut qc: *mut ata_queued_cmd = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
let mut thaw: bool = false;
qc = __ata_qc_from_tag(ap, unsafe { (*ap).link.active_tag });
if !qc.is_null() && (unsafe { (*qc).flags }) & 65536 == 0 {
qc = core::ptr::null_mut::<ata_queued_cmd>();
}
// reset PIO HSM and stop DMA engine
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
let __v3: bool = !qc.is_null() && ata_is_dma(unsafe { (*qc).tf.protocol });
if __v3 {
let host_stat: u8 = unsafe { unsafe { (*(*ap).ops).bmdma_status }.unwrap()(ap) };
trace_ata_bmdma_status(ap, host_stat as u32);
// BMDMA controllers indicate host bus error by
// setting DMA_ERR bit and timing out.  As it wasn't
// really a timeout event, adjust error mask and
// cancel frozen state.
if (unsafe { (*qc).err_mask }) == 4 && (host_stat as i32) & 2 != 0 {
unsafe {
(*qc).err_mask = 32;
}
thaw = true;
}
trace_ata_bmdma_stop(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe { unsafe { (*(*ap).ops).bmdma_stop }.unwrap()(qc) };
// if we're gonna thaw, make sure IRQ is clear
if thaw {
unsafe { unsafe { (*(*ap).ops).sff_check_status }.unwrap()(ap) };
if (unsafe { (*(*ap).ops).sff_irq_clear }) != None {
unsafe { unsafe { (*(*ap).ops).sff_irq_clear }.unwrap()(ap) };
}
}
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
if thaw {
unsafe { ata_eh_thaw_port(ap) };
}
ata_sff_error_handler(ap);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_error_handler_938: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_bmdma_error_handler)) };

/// ata_bmdma_post_internal_cmd - Stock post_internal_cmd for BMDMA
/// @qc: internal command to clean up
///
/// LOCKING:
/// Kernel thread context (may sleep)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_post_internal_cmd")]
extern "C" fn ata_bmdma_post_internal_cmd(qc: *mut ata_queued_cmd) {
let ap: *mut ata_port = unsafe { (*qc).ap };
let mut flags: u64 = 0;
if ata_is_dma(unsafe { (*qc).tf.protocol }) {
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
trace_ata_bmdma_stop(ap, (unsafe { core::ptr::addr_of_mut!((*qc).tf) }) as *const ata_taskfile, unsafe { (*qc).tag });
unsafe { unsafe { (*(*ap).ops).bmdma_stop }.unwrap()(qc) };
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_post_internal_cmd_939: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd)>, *mut ()>(Some(ata_bmdma_post_internal_cmd)) };

/// ata_bmdma_irq_clear - Clear PCI IDE BMDMA interrupt.
/// @ap: Port associated with this ATA transaction.
///
/// Clear interrupt and error flags in DMA status register.
///
/// May be used as the irq_clear() entry in ata_port_operations.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_irq_clear")]
extern "C" fn ata_bmdma_irq_clear(ap: *mut ata_port) {
let mmio: *mut () = unsafe { (*ap).ioaddr.bmdma_addr };
if mmio.is_null() {
return;
}
unsafe { iowrite8(ioread8(((mmio as *const u8).wrapping_offset(2 as isize) as *mut ()) as *const ()) as u8, (mmio as *const u8).wrapping_offset(2 as isize) as *mut ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_irq_clear_940: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port)>, *mut ()>(Some(ata_bmdma_irq_clear)) };

/// ata_bmdma_setup - Set up PCI IDE BMDMA transaction
/// @qc: Info associated with this ATA transaction.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_setup")]
extern "C" fn ata_bmdma_setup(qc: *mut ata_queued_cmd) {
let ap: *mut ata_port = unsafe { (*qc).ap };
let rw: u32 = ((unsafe { (*qc).tf.flags }) & 8) as u32;
let mut dmactl: u8 = 0;
// load PRD table addr.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("mfence", options(nostack, att_syntax));
}
if true {
break;
}
}
// make sure PRD table writes are visible to controller
unsafe { iowrite32((*ap).bmdma_prd_dma as u32, ((*ap).ioaddr.bmdma_addr as *const u8).wrapping_offset(4 as isize) as *mut ()) };
// specify data direction, triple-check start bit is clear
dmactl = (unsafe { ioread8((((*ap).ioaddr.bmdma_addr as *const u8).wrapping_offset(0 as isize) as *mut ()) as *const ()) }) as u8;
let __v3: u8 = ((dmactl as i32) & !(8 | 1)) as u8;
dmactl = __v3;
if rw == 0 {
dmactl = ((dmactl as i32) | 8) as u8;
}
unsafe { iowrite8(dmactl, ((*ap).ioaddr.bmdma_addr as *const u8).wrapping_offset(0 as isize) as *mut ()) };
// issue r/w command
unsafe { unsafe { (*(*ap).ops).sff_exec_command }.unwrap()(ap, core::ptr::addr_of_mut!((*qc).tf) as *const ata_taskfile) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_setup_941: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd)>, *mut ()>(Some(ata_bmdma_setup)) };

/// ata_bmdma_start - Start a PCI IDE BMDMA transaction
/// @qc: Info associated with this ATA transaction.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_start")]
extern "C" fn ata_bmdma_start(qc: *mut ata_queued_cmd) {
let ap: *mut ata_port = unsafe { (*qc).ap };
let mut dmactl: u8 = 0;
// start host DMA transaction
dmactl = (unsafe { ioread8((((*ap).ioaddr.bmdma_addr as *const u8).wrapping_offset(0 as isize) as *mut ()) as *const ()) }) as u8;
unsafe { iowrite8(((dmactl as i32) | 1) as u8, ((*ap).ioaddr.bmdma_addr as *const u8).wrapping_offset(0 as isize) as *mut ()) };
// Strictly, one may wish to issue an ioread8() here, to
// flush the mmio write.  However, control also passes
// to the hardware at this point, and it will interrupt
// us when we are to resume control.  So, in effect,
// we don't care when the mmio write flushes.
// Further, a read of the DMA status register _immediately_
// following the write may not be what certain flaky hardware
// is expected, so I think it is best to not add a readb()
// without first all the MMIO ATA cards/mobos.
// Or maybe I'm just being paranoid.
//
// FIXME: The posting of this write means I/O starts are
// unnecessarily delayed for MMIO
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_start_942: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd)>, *mut ()>(Some(ata_bmdma_start)) };

/// ata_bmdma_stop - Stop PCI IDE BMDMA transfer
/// @qc: Command we are ending DMA for
///
/// Clears the ATA_DMA_START flag in the dma control register
///
/// May be used as the bmdma_stop() entry in ata_port_operations.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_stop")]
extern "C" fn ata_bmdma_stop(qc: *mut ata_queued_cmd) {
let ap: *mut ata_port = unsafe { (*qc).ap };
let mmio: *mut () = unsafe { (*ap).ioaddr.bmdma_addr };
// clear start/stop bit
unsafe { iowrite8((ioread8(((mmio as *const u8).wrapping_offset(0 as isize) as *mut ()) as *const ()) & 4294967294u32) as u8, (mmio as *const u8).wrapping_offset(0 as isize) as *mut ()) };
// one-PIO-cycle guaranteed wait, per spec, for HDMA1:0 transition
ata_sff_dma_pause(ap);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_stop_943: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_queued_cmd)>, *mut ()>(Some(ata_bmdma_stop)) };

/// ata_bmdma_status - Read PCI IDE BMDMA status
/// @ap: Port associated with this ATA transaction.
///
/// Read and return BMDMA status register.
///
/// May be used as the bmdma_status() entry in ata_port_operations.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_status")]
extern "C" fn ata_bmdma_status(ap: *mut ata_port) -> u8 {
(unsafe { ioread8((((*ap).ioaddr.bmdma_addr as *const u8).wrapping_offset(2 as isize) as *mut ()) as *const ()) }) as u8
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_status_944: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port) -> u8>, *mut ()>(Some(ata_bmdma_status)) };

/// ata_bmdma_port_start - Set port up for bmdma.
/// @ap: Port to initialize
///
/// Called just after data structures for each port are
/// initialized.  Allocates space for PRD table.
///
/// May be used as the port_start() entry in ata_port_operations.
///
/// LOCKING:
/// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_port_start")]
extern "C" fn ata_bmdma_port_start(ap: *mut ata_port) -> i32 {
if (unsafe { (*ap).mwdma_mask }) != 0 || (unsafe { (*ap).udma_mask }) != 0 {
unsafe {
(*ap).bmdma_prd = dmam_alloc_coherent((*(*ap).host).dev, 2048u64, core::ptr::addr_of_mut!((*ap).bmdma_prd_dma), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32)) as *mut ata_bmdma_prd;
}
if unsafe { (*ap).bmdma_prd }.is_null() {
return 12i32.wrapping_neg();
}
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_port_start_945: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port) -> i32>, *mut ()>(Some(ata_bmdma_port_start)) };

/// ata_bmdma_port_start32 - Set port up for dma.
/// @ap: Port to initialize
///
/// Called just after data structures for each port are
/// initialized.  Enables 32bit PIO and allocates space for PRD
/// table.
///
/// May be used as the port_start() entry in ata_port_operations for
/// devices that are capable of 32bit PIO.
///
/// LOCKING:
/// Inherited from caller.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_bmdma_port_start32")]
extern "C" fn ata_bmdma_port_start32(ap: *mut ata_port) -> i32 {
unsafe {
(*ap).pflags = (*ap).pflags | 3145728u32;
}
ata_bmdma_port_start(ap)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_bmdma_port_start32_946: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port) -> i32>, *mut ()>(Some(ata_bmdma_port_start32)) };

/// ata_pci_bmdma_clear_simplex -	attempt to kick device out of simplex
/// @pdev: PCI device
///
/// Some PCI ATA devices report simplex mode but in fact can be told to
/// enter non simplex mode. This implements the necessary logic to
/// perform the task on such devices. Calling it on other devices will
/// have -undefined- behaviour.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_pci_bmdma_clear_simplex")]
extern "C" fn ata_pci_bmdma_clear_simplex(pdev: *mut pci_dev) -> i32 {
let bmdma: u64 = pci_resource_start!(pdev, 4i32);
let mut simplex: u8 = 0;
if bmdma == 0 {
return 2i32.wrapping_neg();
}
simplex = __inb(bmdma.wrapping_add(2) as u16);
__outb(((simplex as i32) & 96) as u8, bmdma.wrapping_add(2) as u16);
simplex = __inb(bmdma.wrapping_add(2) as u16);
if (simplex as i32) & 128 != 0 {
return 95i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_pci_bmdma_clear_simplex_947: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev) -> i32>, *mut ()>(Some(ata_pci_bmdma_clear_simplex)) };

fn ata_bmdma_nodma(host: *mut ata_host, reason: *const u8) {
while false {
}
unsafe { _dev_err((*host).dev as *const device, c"BMDMA: %s, falling back to PIO\n".as_ptr() as *const u8, reason) };
for i in 0i32..2i32 {
unsafe {
(*(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize))).mwdma_mask = 0;
}
unsafe {
(*(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize))).udma_mask = 0;
}
}
}

/// ata_pci_bmdma_init - acquire PCI BMDMA resources and init ATA host
/// @host: target ATA host
///
/// Acquire PCI BMDMA resources and initialize @host accordingly.
///
/// LOCKING:
/// Inherited from calling layer (may sleep).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_pci_bmdma_init")]
extern "C" fn ata_pci_bmdma_init(host: *mut ata_host) {
let gdev: *mut device = unsafe { (*host).dev };
let pdev: *mut pci_dev = to_pci_dev!(gdev);
let mut rc: i32 = 0;
// No BAR4 allocation: No DMA
if pci_resource_start!(pdev, 4i32) == 0 {
ata_bmdma_nodma(host, c"BAR4 is zero".as_ptr() as *const u8);
return;
}
// Some controllers require BMDMA region to be initialized
// even if DMA is not in use to clear IRQ status via
// ->sff_irq_clear method.  Try to initialize bmdma_addr
// regardless of dma masks.
rc = dma_set_mask_and_coherent(unsafe { core::ptr::addr_of_mut!((*pdev).dev) }, 4294967295u64);
if rc != 0 {
ata_bmdma_nodma(host, c"failed to set dma mask".as_ptr() as *const u8);
}
// request and iomap DMA region
rc = unsafe { pcim_iomap_regions(pdev, 16i32, dev_driver_string(gdev as *const device)) };
if rc != 0 {
ata_bmdma_nodma(host, c"failed to request/iomap BAR4".as_ptr() as *const u8);
return;
}
unsafe {
(*host).iomap = pcim_iomap_table(pdev);
}
for i in 0i32..2i32 {
let ap: *mut ata_port = unsafe { *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(i as isize) };
let bmdma: *mut () = (unsafe { (*unsafe { (*host).iomap }.wrapping_offset(4 as isize) as *const u8).wrapping_offset(8i32.wrapping_mul(i) as isize) }) as *mut ();
if ata_port_is_dummy(ap) == 0 {
unsafe {
(*ap).ioaddr.bmdma_addr = bmdma;
}
let __v2: bool = (unsafe { (*ap).flags }) & 32768 == 0 && (unsafe { ioread8(((bmdma as *const u8).wrapping_offset(2 as isize) as *mut ()) as *const ()) }) & 128 != 0;
if __v2 {
unsafe {
(*host).flags = (*host).flags | 1;
}
}
unsafe { ata_port_desc(ap, c"bmdma 0x%llx".as_ptr() as *const u8, pci_resource_start!(pdev, 4i32).wrapping_add(8i32.wrapping_mul(i) as u64)) };
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_pci_bmdma_init_948: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_host)>, *mut ()>(Some(ata_pci_bmdma_init)) };

/// ata_pci_bmdma_prepare_host - helper to prepare PCI BMDMA ATA host
/// @pdev: target PCI device
/// @ppi: array of port_info, must be enough for two ports
/// @r_host: out argument for the initialized ATA host
///
/// Helper to allocate BMDMA ATA host for @pdev, acquire all PCI
/// resources and initialize it accordingly in one go.
///
/// LOCKING:
/// Inherited from calling layer (may sleep).
///
/// RETURNS:
/// 0 on success, -errno otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_pci_bmdma_prepare_host")]
extern "C" fn ata_pci_bmdma_prepare_host(pdev: *mut pci_dev, ppi: *const *const ata_port_info, r_host: *mut *mut ata_host) -> i32 {
let rc: i32 = ata_pci_sff_prepare_host(pdev, ppi, r_host);
if rc != 0 {
return rc;
}
ata_pci_bmdma_init(unsafe { *r_host });
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_pci_bmdma_prepare_host_949: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *const *const ata_port_info, *mut *mut ata_host) -> i32>, *mut ()>(Some(ata_pci_bmdma_prepare_host)) };

/// ata_pci_bmdma_init_one - Initialize/register BMDMA PCI IDE controller
/// @pdev: Controller to be initialized
/// @ppi: array of port_info, must be enough for two ports
/// @sht: scsi_host_template to use when registering the host
/// @host_priv: host private_data
/// @hflags: host flags
///
/// This function is similar to ata_pci_sff_init_one() but also
/// takes care of BMDMA initialization.
///
/// LOCKING:
/// Inherited from PCI layer (may sleep).
///
/// RETURNS:
/// Zero on success, negative on errno-based value on error.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_sff.ata_pci_bmdma_init_one")]
extern "C" fn ata_pci_bmdma_init_one(pdev: *mut pci_dev, ppi: *const *const ata_port_info, sht: *const scsi_host_template, host_priv: *mut (), hflags: i32) -> i32 {
ata_pci_init_one(pdev, ppi, sht, host_priv, hflags, true)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_pci_bmdma_init_one_950: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut pci_dev, *const *const ata_port_info, *const scsi_host_template, *mut (), i32) -> i32>, *mut ()>(Some(ata_pci_bmdma_init_one)) };

/// ata_sff_port_init - Initialize SFF/BMDMA ATA port
/// @ap: Port to initialize
///
/// Called on port allocation to initialize SFF/BMDMA specific
/// fields.
///
/// LOCKING:
/// None.
#[unsafe(no_mangle)]
extern "C" fn ata_sff_port_init(ap: *mut ata_port) {
let mut __slate_compound_0: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
'__slate_break_3: loop {
'__slate_break_2: loop {
'__slate_break_1: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*ap).sff_pio_task)).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*ap).sff_pio_task)).work)).data = *{
__slate_compound_0 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*ap).sff_pio_task)).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*ap).sff_pio_task)).work)).func = Some(ata_sff_pio_task);
}
if true {
break;
}
}
if true {
break;
}
}
__timer_init!(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*ap).sff_pio_task)).timer) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
unsafe {
(*ap).ctl = 8;
}
unsafe {
(*ap).last_ctl = 255;
}
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn ata_sff_init() -> i32 {
let mut __v0: *mut workqueue_struct;
let mut __v1: *mut workqueue_struct;
let mut _res: *mut workqueue_struct = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = unsafe { alloc_workqueue_noprof(c"ata_sff".as_ptr() as *const u8, 264u32, 2048i32) };
alloc_tag_restore!();
} else {
_res = unsafe { alloc_workqueue_noprof(c"ata_sff".as_ptr() as *const u8, 264u32, 2048i32) };
}
__v1 = _res;
__v0 = __v1;
unsafe {
ata_sff_wq = __v0;
}
if unsafe { ata_sff_wq }.is_null() {
return 12i32.wrapping_neg();
}
0
}

#[unsafe(no_mangle)]
extern "C" fn ata_sff_exit() {
unsafe { destroy_workqueue(ata_sff_wq) };
}

static __slate_asm_string_ata_bmdma_qc_issue_1: [u8; 1] = *b"\0";

static __slate_asm_string_ata_bmdma_qc_issue_12: [u8; 1] = *b"\0";

static __slate_asm_string_ata_bmdma_qc_issue_13: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_bmdma_qc_issue_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_bmdma_qc_issue_7: [u8; 1] = *b"\0";

static __slate_asm_string_ata_bmdma_qc_issue_8: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_pio_sector_7: [u8; 1] = *b"\0";

static __slate_asm_string_ata_pio_sector_8: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_pio_sectors_1: [u8; 1] = *b"\0";

static __slate_asm_string_ata_pio_sectors_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_dma_pause_0: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_dma_pause_1: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_hsm_move_55: [u8; 50] = *b"ata%d: SFF host state machine in invalid state %d\0";

static __slate_asm_string_ata_sff_hsm_move_56: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_hsm_move_57: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_hsm_move_58: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_hsm_move_59: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_hsm_move_60: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_hsm_move_61: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_hsm_move_62: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_lost_interrupt_1: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_lost_interrupt_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_pio_task_12: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_pio_task_13: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_pio_task_14: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_pio_task_15: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_queue_pio_task_1: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_queue_pio_task_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_tf_load_1: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_tf_load_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_ata_sff_tf_read_1: [u8; 1] = *b"\0";

static __slate_asm_string_ata_sff_tf_read_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_atapi_send_cdb_1: [u8; 1] = *b"\0";

static __slate_asm_string_atapi_send_cdb_2: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_atapi_send_cdb_6: [u8; 1] = *b"\0";

static __slate_asm_string_atapi_send_cdb_7: [u8; 54] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-sff.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

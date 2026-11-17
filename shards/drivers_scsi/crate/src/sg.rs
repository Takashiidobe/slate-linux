
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
//!  History:
//!  Started: Aug 9 by Lawrence Foard (entropy@world.std.com),
//!           to allow user process control of SCSI devices.
//!  Development Sponsored by Killy Corp. NY NY
//!
//! Original driver (sg.c):
//!        Copyright (C) 1992 Lawrence Foard
//! Version 2 and 3 extensions to driver:
//!        Copyright (C) 1998 - 2014 Douglas Gilbert
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
//! Force a compilation error if condition is true, but also produce a
//! result (of value 0 and type int), so the expression can be used
//! e.g. in a structure initializer (or where-ever else comma expressions
//! aren't permitted).
//!
//! Take an error message as an optional second argument. If omitted,
//! default to the stringification of the tested expression.
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! See linux/instrumentation.h
//! 64 bit
//! unused, was: #define EX_TYPE_COPY	 4
//! Begin/end of an instrumentation safe region
//! inhibit tail-call optimization
//! Call a function once. Similar to DO_ONCE(), but does not use jump label
//! patching via static keys.
//! These are for everybody (although not all archs will actually
//! discard it in modules)
//! warning conditions
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! # chars in a file name
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
//! label
//! Prevent speculative execution past this barrier.
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
//! writeable
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! No ioctl command
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
//! ARRAY_SIZE - get the number of elements in array @arr
//! @arr: array to be sized
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
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
//! Used to create numbers.
//!
//! NOTE: _IOW means userland is writing and kernel is reading. _IOR
//! means userland is reading and kernel is writing.
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
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! Used in tsk->__state:
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
//! Desired design of maximum size and alignment (see RFC2553)
//! return the ->next pointer of a list_head in an rcu safe
//! way, we must not access it directly
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
//! not fcntl
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! get max sectors per request (ll_rw_blk.c)
//! file is open for writing
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
//! number of env pointers
//! Generic info of form tag = "info"
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
//! These are specified by iBCS2
//! Epoll event masks
//! Generate SD flag indexes
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! iterate over multi-page bvec
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! See the comment in asm-generic/io.h about ioremap_np().
//! The Well Known LUNS (SAM-3) in our int representation of a LUN
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! identify no tag in use
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
// net/core/page_pool.c
//
// page_pool uses additional free bits within this value to store data, see the
// definition of PP_DMA_INDEX_MASK in mm.h
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
// normal but significant condition
// informational
// some user interfaces are
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
// pr_info - Print an info-level message
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_INFO loglevel. It uses pr_fmt() to
// generate the format string.
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
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
// Math argument out of domain of func
// Math result not representable
// Invalid system call number
// Value too large for defined data type
// Message too long
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
// max_t - return maximum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
// used to decode ioctl numbers..
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
// disable sas during sighandling
// For trivial one-depth nesting of a lock-class, the following
// global define can be used. (Subsystems with multiple levels
// of nesting should define their own lock-nesting subclasses.)
// Writer mask
// Remapping rwlock architecture specific functions to the corresponding
// queued rwlock functions.
// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
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
// used to be O_SYNC, see below
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
// per-IO O_SYNC
// per-IO, return -EAGAIN if operation would block
// per-IO O_APPEND
// per-IO negation of O_APPEND
// Atomic Write
// buffered IO that drops the cache after reading or writing data
// mask of flags supported by the kernel
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// Generate SD flag bits
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// Used to obtain PUN and LUN info.  Conflicts with CDROMAUDIOBUFSIZ
// Used to obtain the host number of a device.
// Used to obtain the bus number for a device
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
/// @a is a power of 2 value
macro_rules! ALIGN__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!($x, $a)
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

/// BUILD_BUG_ON_INVALID() permits the compiler to check the validity of the
/// expression but avoids the generation of any code, even if that expression
/// has side-effects.
macro_rules! BUILD_BUG_ON_INVALID {
() => {
};
}

macro_rules! COMMAND_SIZE {
($opcode:expr) => {
unsafe { scsi_command_size_tbl[((($opcode) as i32) >> 5i32 & 7) as usize] }
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

macro_rules! MINOR {
($dev:expr) => {
($dev) & 1048575u32
};
}

macro_rules! MKDEV {
($ma:expr, $mi:expr) => {
((($ma) << 20i32) as u32) | $mi
};
}

macro_rules! MKDEV__2 {
($ma:expr, $mi:expr) => {
($ma) << 20i32 | $mi
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

macro_rules! RCU_LOCKDEP_WARN__2 {
($c:expr) => {
while false && $c {
}
};
}

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
};
}

macro_rules! SCSI_CHECK_LOGGING {
() => {
while false {
}
};
}

macro_rules! SCSI_LOG_TIMEOUT {
() => {
SCSI_CHECK_LOGGING!();
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VM_BUG_ON_FOLIO {
() => {
VM_BUG_ON!();
};
}

macro_rules! VM_BUG_ON {
() => {
BUILD_BUG_ON_INVALID!();
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

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as i32).wrapping_sub(1))
};
}

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
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

macro_rules! ___wait_is_interruptible {
($state:expr) => {
false || ($state) & (1 | 256) != 0
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

macro_rules! __chk_user_ptr {
() => {
};
}

macro_rules! __cmp_once__3 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once__2 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = $y;
__v = if __v2 < __v3 { __v2 } else { __v3 };
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
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
macro_rules! __native_word__6 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

macro_rules! __ratelimit {
($state:expr) => {
unsafe { ___ratelimit($state, c"sg_write".as_ptr() as *const u8) }
};
}

macro_rules! __rcu_dereference_protected__2 {
($p:expr, $c:expr) => {
{
let mut __v: *const cred;
RCU_LOCKDEP_WARN__2!(($c) == 0);
__v = $p;
__v
}
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release__2 {
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

macro_rules! alloc_hooks__4 {
($_do_alloc:expr) => {
{
let mut __v: *mut page;
let mut __v2: *mut page;
let mut __v3: *mut page = unsafe { core::mem::zeroed() };
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
macro_rules! container_of__14 {
($ptr:expr) => {
{
let mut __v: *mut sg_request;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__47!() as isize)) }) as *mut ()) as *mut sg_request;
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
macro_rules! container_of__17 {
($ptr:expr) => {
{
let mut __v: *mut sg_fd;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__37!() as isize)) }) as *mut ()) as *mut sg_fd;
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
macro_rules! container_of__23 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__43!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__25 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__44!($type) as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__26 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__19!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

/// current_real_cred - Access the current task's objective credentials
///
/// Access the objective credentials of the current task.  RCU-safe,
/// since nobody else can modify it.
macro_rules! current_real_cred {
() => {
rcu_dereference_protected__2!(unsafe { (*get_current()).real_cred }, 1i32)
};
}

macro_rules! current_thread_info {
() => {
get_current() as *mut thread_info
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
macro_rules! data_race {
($expr:expr) => {
{
let mut __v: u32;
__kcsan_disable_current();
let __v2: u32 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
}
};
}

/// Obsolete driver_byte() declaration
macro_rules! driver_byte {
($result:expr) => {
($result) >> 24i32 & 255
};
}

macro_rules! host_byte {
($result:expr) => {
($result) >> 16i32 & 255
};
}

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2__2 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub($n.leading_zeros() as i32) } } else { if (core::mem::size_of::<u64>() as u64) <= 4 { __ilog2_u32(($n) as u32) } else { __ilog2_u64($n) } }
};
}

macro_rules! in_ia32_syscall {
() => {
{
let __v: bool = IS_ENABLED__2!() != 0 && (unsafe { (*current_thread_info!()).status }) & 2 != 0;
__v
}
};
}

/// instrument_get_user() - add instrumentation to get_user()-like macros
/// @to: destination variable, may not be address-taken
///
/// get_user() and friends are fragile, so it may depend on the implementation
/// whether the instrumentation happens before or after the data is copied from
/// the userspace.
macro_rules! instrument_get_user__2 {
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
macro_rules! instrument_put_user {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<i32>() as u64, 0u64);
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

macro_rules! likely__2 {
($x:expr) => {
(($x) != 0) as i64
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
macro_rules! list_entry__7 {
($ptr:expr) => {
container_of__17!($ptr)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__14 {
($ptr:expr, $type:ty) => {
container_of__26!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__13 {
($ptr:expr) => {
container_of__14!($ptr)
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
macro_rules! list_first_entry__3 {
($ptr:expr) => {
list_entry__13!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__5 {
($ptr:expr) => {
list_entry__7!(unsafe { (*$ptr).next })
};
}

/// list_first_entry - get the first element from a list
/// @ptr:	the list head to take the element from.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
///
/// Note, that list is expected to be not empty.
macro_rules! list_first_entry__10 {
($ptr:expr, $type:ty) => {
list_entry__14!(unsafe { (*$ptr).next }, $type)
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__5 {
($pos:expr, $member:ident) => {
list_entry__7!(unsafe { (*$pos).$member.next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__3 {
($pos:expr, $member:ident) => {
list_entry__13!(unsafe { (*$pos).$member.next })
};
}

macro_rules! lockdep_assert_held_write {
($l:expr) => {
__assume_ctx_lock!($l);
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

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__3!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__3 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__2!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

/// Calculate "x * n / d" without unnecessary overflow or loss of precision.
macro_rules! mult_frac {
($x:expr, $n:expr, $d:expr) => {
{
let mut __v: i32;
let __v2: i32 = $x;
let __v3: i32 = $n;
let __v4: i32 = $d;
let __v5: i32 = __v2 / __v4;
let __v6: i32 = __v2 % __v4;
__v = __v5.wrapping_mul(__v3).wrapping_add(__v6.wrapping_mul(__v3) / __v4);
__v
}
};
}

/// Calculate "x * n / d" without unnecessary overflow or loss of precision.
macro_rules! mult_frac__2 {
($x:expr, $n:expr, $d:expr) => {
{
let mut __v: i64;
let __v2: i64 = $x;
let __v3: i32 = $n;
let __v4: i32 = $d;
let __v5: i64 = __v2 / (__v4 as i64);
let __v6: i64 = __v2 % (__v4 as i64);
__v = __v5.wrapping_mul(__v3 as i64).wrapping_add(__v6.wrapping_mul(__v3 as i64) / (__v4 as i64));
__v
}
};
}

macro_rules! offsetof__8 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, ew.work) as u64
};
}

macro_rules! offsetof {
() => {
40u64
};
}

macro_rules! offsetof__19 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, entry) as u64
};
}

macro_rules! offsetof__35 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, usr_ptr) as u64
};
}

macro_rules! offsetof__37 {
() => {
core::mem::offset_of!(sg_fd, sfd_siblings) as u64
};
}

macro_rules! offsetof__43 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, d_ref) as u64
};
}

macro_rules! offsetof__44 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, f_ref) as u64
};
}

macro_rules! offsetof__6 {
() => {
456u64
};
}

macro_rules! offsetof__47 {
() => {
core::mem::offset_of!(sg_request, entry) as u64
};
}

macro_rules! page_address__2 {
($page:expr) => {
lowmem_page_address(($page) as *const page)
};
}

/// page_folio - Converts from page to folio.
/// @p: The page.
///
/// Every page is part of a folio.  This function cannot be called on a
/// NULL pointer.
///
/// Context: No reference, nor lock is required on @page.  If the caller
/// does not hold a reference, this call may race with a folio split, so
/// it should re-check the folio still contains this page after gaining
/// a reference on the folio.
/// Return: The folio which contains this page.
macro_rules! page_folio {
($p:expr) => {
_compound_head(($p) as *const page) as *mut folio
};
}

macro_rules! page_ref_tracepoint_active {
() => {
0i32
};
}

macro_rules! page_to_virt {
($x:expr) => {
__va!(PFN_PHYS!(((unsafe { $x.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64))
};
}

macro_rules! preempt_count_dec_and_test {
() => {
__preempt_count_dec_and_test()
};
}

macro_rules! proc_create_seq {
($name:expr, $mode:expr, $parent:expr, $ops:expr) => {
unsafe { proc_create_seq_private(($name) as *const u8, ($mode) as u16, $parent, $ops, 0u32, core::ptr::null_mut::<()>()) };
};
}

macro_rules! proc_create_single {
($name:expr, $mode:expr, $parent:expr, $show:expr) => {
unsafe { proc_create_single_data(($name) as *const u8, ($mode) as u16, $parent, $show, core::ptr::null_mut::<()>()) };
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
macro_rules! rcu_dereference_protected__2 {
($p:expr, $c:expr) => {
__rcu_dereference_protected__2!($p, $c)
};
}

macro_rules! read_lock {
($lock:expr) => {
unsafe { _raw_read_lock($lock) };
};
}

macro_rules! read_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
$flags = unsafe { _raw_read_lock_irqsave($lock) };
if true {
break;
}
}
};
}

macro_rules! read_unlock {
($lock:expr) => {
unsafe { _raw_read_unlock($lock) };
};
}

macro_rules! read_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_read_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

macro_rules! scsi_unregister_interface {
($intf:expr) => {
unsafe { class_interface_unregister($intf) };
};
}

/// Obsolete status_byte() declaration
macro_rules! sg_status_byte {
($result:expr) => {
($result) >> 1i32 & 127
};
}

macro_rules! to_scsi_device {
($d:expr) => {
{
let mut __v: *mut scsi_device;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof__6!() as isize)) }) as *mut ()) as *mut scsi_device;
__v
}
};
}

macro_rules! to_scsi_target {
($d:expr) => {
{
let mut __v: *mut scsi_target;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof!() as isize)) }) as *mut ()) as *mut scsi_target;
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

macro_rules! unlikely__5 {
($x:expr) => {
(($x) != 0) as i64
};
}

macro_rules! wake_up_interruptible {
($x:expr) => {
unsafe { __wake_up($x, 1u32, 1i32, core::ptr::null_mut::<()>()) };
};
}

macro_rules! wake_up_interruptible_all {
($x:expr) => {
unsafe { __wake_up($x, 1u32, 0i32, core::ptr::null_mut::<()>()) };
};
}

macro_rules! write_lock_irq {
($lock:expr) => {
unsafe { _raw_write_lock_irq($lock) };
};
}

macro_rules! write_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
$flags = unsafe { _raw_write_lock_irqsave($lock) };
if true {
break;
}
}
};
}

macro_rules! write_unlock_irq {
($lock:expr) => {
unsafe { _raw_write_unlock_irq($lock) };
};
}

macro_rules! write_unlock_irqrestore {
($lock:expr, $flags:expr) => {
loop {
typecheck!(u64);
unsafe { _raw_write_unlock_irqrestore($lock, $flags) };
if true {
break;
}
}
};
}

unsafe extern "C" {
    static mut kmsg_fops: file_operations;
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut __SCK__might_resched: static_call_key;
    static mut __invalid_size_argument_for_IOC: u32;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut radix_tree_preloads: radix_tree_preload;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut param_ops_int: kernel_param_ops;
    static mut scsi_command_size_tbl: [u8; 8];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __compiletime_assert_0();
    fn __compiletime_assert_1();
    fn __compiletime_assert_2();
    fn __compiletime_assert_5();
    fn __compiletime_assert_6();
    fn __bad_size_call_parameter();
    fn __compiletime_assert_29();
    fn __xadd_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn kstrtoint(arg0: *const u8, arg1: u32, arg2: *mut i32) -> i32;
    fn kstrtoul_from_user(arg0: *const u8, arg1: u64, arg2: u32, arg3: *mut u64) -> i32;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn __cond_resched() -> i32;
    fn __SCT__might_resched() -> i32;
    fn _raw_read_lock(arg0: *mut rwlock);
    fn _raw_write_lock_irq(arg0: *mut rwlock);
    fn _raw_read_lock_irqsave(arg0: *mut rwlock) -> u64;
    fn _raw_write_lock_irqsave(arg0: *mut rwlock) -> u64;
    fn _raw_read_unlock(arg0: *mut rwlock);
    fn _raw_write_unlock_irq(arg0: *mut rwlock);
    fn _raw_read_unlock_irqrestore(arg0: *mut rwlock, arg1: u64);
    fn _raw_write_unlock_irqrestore(arg0: *mut rwlock, arg1: u64);
    fn schedule();
    fn __init_waitqueue_head(arg0: *mut wait_queue_head, arg1: *const u8, arg2: *mut lock_class_key);
    fn __wake_up(arg0: *mut wait_queue_head, arg1: u32, arg2: i32, arg3: *mut ()) -> i32;
    fn init_wait_entry(arg0: *mut wait_queue_entry, arg1: i32);
    fn prepare_to_wait_event(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry, arg2: i32) -> i64;
    fn finish_wait(arg0: *mut wait_queue_head, arg1: *mut wait_queue_entry);
    fn mutex_init_generic(arg0: *mut mutex);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn refcount_warn_saturate(arg0: *mut refcount_struct, arg1: u32);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn __compiletime_assert_187();
    fn alloc_pages_noprof(arg0: u32, arg1: u32) -> *mut page;
    fn __free_pages(arg0: *mut page, arg1: u32);
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn __vma_start_write(arg0: *mut vm_area_struct, arg1: i32) -> i32;
    fn _copy_from_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn _copy_to_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn capable(arg0: i32) -> bool;
    fn __task_pid_nr_ns(arg0: *mut task_struct, arg1: u32, arg2: *mut pid_namespace) -> i32;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn radix_tree_preload(arg0: u32) -> i32;
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn fasync_helper(arg0: i32, arg1: *mut file, arg2: i32, arg3: *mut *mut fasync_struct) -> i32;
    fn kill_fasync(arg0: *mut *mut fasync_struct, arg1: i32, arg2: i32);
    fn compat_ptr_ioctl(arg0: *mut file, arg1: u32, arg2: u64) -> i64;
    fn register_chrdev_region(arg0: u32, arg1: u32, arg2: *const u8) -> i32;
    fn unregister_chrdev_region(arg0: u32, arg1: u32);
    fn nonseekable_open(arg0: *mut inode, arg1: *mut file) -> i32;
    fn idr_preload(arg0: u32);
    fn idr_alloc(arg0: *mut idr, arg1: *mut (), arg2: i32, arg3: i32, arg4: u32) -> i32;
    fn idr_remove(arg0: *mut idr, arg1: u64) -> *mut ();
    fn idr_find(arg0: *const idr, arg1: u64) -> *mut ();
    fn idr_for_each(arg0: *const idr, arg1: Option<unsafe extern "C" fn(i32, *mut (), *mut ()) -> i32>, arg2: *mut ()) -> i32;
    fn idr_destroy(arg0: *mut idr);
    fn sysfs_create_link(arg0: *mut kobject, arg1: *mut kobject, arg2: *const u8) -> i32;
    fn sysfs_remove_link(arg0: *mut kobject, arg1: *const u8);
    fn param_get_int(arg0: *mut u8, arg1: *const kernel_param) -> i32;
    fn __modver_version_show(arg0: *const module_attribute, arg1: *mut module_kobject, arg2: *mut u8) -> i64;
    fn __module_get(arg0: *mut module);
    fn module_put(arg0: *mut module);
    fn class_register(arg0: *const class) -> i32;
    fn class_unregister(arg0: *const class);
    fn class_interface_unregister(arg0: *mut class_interface);
    fn device_create(arg0: *const class, arg1: *mut device, arg2: u32, arg3: *mut (), arg4: *const u8, ...) -> *mut device;
    fn device_destroy(arg0: *const class, arg1: u32);
    fn cdev_alloc() -> *mut cdev;
    fn cdev_add(arg0: *mut cdev, arg1: u32, arg2: u32) -> i32;
    fn cdev_del(arg0: *mut cdev);
    fn seq_read(arg0: *mut file, arg1: *mut u8, arg2: u64, arg3: *mut i64) -> i64;
    fn seq_lseek(arg0: *mut file, arg1: i64, arg2: i32) -> i64;
    fn seq_write(arg0: *mut seq_file, arg1: *const (), arg2: u64) -> i32;
    fn seq_printf(arg0: *mut seq_file, arg1: *const u8, ...);
    fn seq_putc(arg0: *mut seq_file, arg1: u8);
    fn __seq_puts(arg0: *mut seq_file, arg1: *const u8);
    #[link_name = "strlen"]
    fn __builtin_strlen(arg0: *const u8) -> u64;
    fn single_open(arg0: *mut file, arg1: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>, arg2: *mut ()) -> i32;
    fn single_release(arg0: *mut inode, arg1: *mut file) -> i32;
    fn sbitmap_weight(arg0: *const sbitmap) -> u32;
    fn blk_get_queue(arg0: *mut request_queue) -> bool;
    fn blk_put_queue(arg0: *mut request_queue);
    fn blk_mq_free_request(arg0: *mut request);
    fn blk_rq_map_user_io(arg0: *mut request, arg1: *mut rq_map_data, arg2: *mut (), arg3: u64, arg4: u32, arg5: bool, arg6: i32, arg7: bool, arg8: i32) -> i32;
    fn blk_rq_unmap_user(arg0: *mut bio) -> i32;
    fn blk_execute_rq_nowait(arg0: *mut request, arg1: bool);
    fn blk_trace_setup(arg0: *mut request_queue, arg1: *mut u8, arg2: u32, arg3: *mut block_device, arg4: *mut u8) -> i32;
    fn blk_trace_startstop(arg0: *mut request_queue, arg1: i32) -> i32;
    fn blk_trace_remove(arg0: *mut request_queue) -> i32;
    fn scsi_normalize_sense(arg0: *const u8, arg1: i32, arg2: *mut scsi_sense_hdr) -> bool;
    fn sdev_prefix_printk(arg0: *const u8, arg1: *const scsi_device, arg2: *const u8, arg3: *const u8, ...);
    fn scsi_device_get(arg0: *mut scsi_device) -> i32;
    fn scsi_device_put(arg0: *mut scsi_device);
    fn scsi_alloc_request(arg0: *mut request_queue, arg1: u32, arg2: u32) -> *mut request;
    fn __scsi_print_sense(arg0: *const scsi_device, arg1: *const u8, arg2: *const u8, arg3: i32);
    fn scsi_register_interface(arg0: *mut class_interface) -> i32;
    fn scsi_block_when_processing_errors(arg0: *mut scsi_device) -> i32;
    fn scsi_template_proc_dir(arg0: *const scsi_host_template) -> *mut proc_dir_entry;
    fn scsi_ioctl_block_when_processing_errors(arg0: *mut scsi_device, arg1: i32, arg2: bool) -> i32;
    fn scsi_ioctl(arg0: *mut scsi_device, arg1: bool, arg2: i32, arg3: *mut ()) -> i32;
    fn get_sg_io_hdr(arg0: *mut sg_io_hdr, arg1: *const ()) -> i32;
    fn put_sg_io_hdr(arg0: *const sg_io_hdr, arg1: *mut ()) -> i32;
    fn scsi_cmd_allowed(arg0: *mut u8, arg1: bool) -> bool;
    fn proc_mkdir(arg0: *const u8, arg1: *mut proc_dir_entry) -> *mut proc_dir_entry;
    fn proc_create_seq_private(arg0: *const u8, arg1: u16, arg2: *mut proc_dir_entry, arg3: *const seq_operations, arg4: u32, arg5: *mut ()) -> *mut proc_dir_entry;
    fn proc_create_single_data(arg0: *const u8, arg1: u16, arg2: *mut proc_dir_entry, arg3: Option<unsafe extern "C" fn(*mut seq_file, *mut ()) -> i32>, arg4: *mut ()) -> *mut proc_dir_entry;
    fn proc_create(arg0: *const u8, arg1: u16, arg2: *mut proc_dir_entry, arg3: *const proc_ops) -> *mut proc_dir_entry;
    fn remove_proc_subtree(arg0: *const u8, arg1: *mut proc_dir_entry) -> i32;
    fn __compiletime_assert_622();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_f217c4720f04f6db(arg0: u64) -> bool;
}

/// Optional: only supported since gcc >= 15, clang >= 19
///
///   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
/// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
/// 2 digits for each component
static mut sg_version_num: i32 = 30536;

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
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
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
struct io_bitmap {
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


fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_0() };
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
unsafe { __compiletime_assert_1() };
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_2() };
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

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_5() };
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

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_6() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
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


#[inline(always)]
fn get_current() -> *mut task_struct {
if IS_ENABLED!() != 0 {
let mut __v0: *mut task_struct;
loop {
if true {
unsafe { __compiletime_assert_29() };
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

fn kmsan_unpoison_memory(address: *const (), size: u64) {
}

fn kmsan_copy_to_user(to: *mut (), from: *const (), to_copy: u64, left: u64) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
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
fn fls(mut x: u32) -> i32 {
let mut r: i32 = 0;
if unsafe { __slate_intrinsic_1c5364fcd9da51fe(x) } {
return if x != 0 { 32i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRL won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before, except that the
// top 32 bits will be cleared.
//
// We cannot do this on 32 bits because at the very least some
// 486 CPUs did not behave this way.
unsafe {
let __t0: u32;
core::arch::asm!("bsrl {1:e},{0:e}", inlateout(reg) (1 as i32).wrapping_neg() as u32 => __t0, in(reg) x as u32, options(pure, nomem, nostack, att_syntax));
r = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
r.wrapping_add(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


#[inline(always)]
fn fls64(mut x: u64) -> i32 {
let mut bitpos: i32 = 1i32.wrapping_neg();
if unsafe { __slate_intrinsic_f217c4720f04f6db(x) } {
return if x != 0 { 64i32.wrapping_sub(x.leading_zeros() as i32) } else { 0 };
}
// AMD64 says BSRQ won't clobber the dest reg if x==0; Intel64 says the
// dest reg is undefined if x==0, but their CPU architect says its
// value is written to set it to the same as before.
unsafe {
let __t0: u32;
core::arch::asm!("bsrq {1:r},{0:r}", inlateout(reg) bitpos as u32 => __t0, in(reg) x as u64, options(pure, nomem, nostack, att_syntax));
bitpos = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
bitpos.wrapping_add(1)
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
core::arch::asm!("32: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 32b - ., 3; .popsection\n# {0}", const 32 as i32, options(nostack, att_syntax));
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
core::arch::asm!("33: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 33b - ., 4; .popsection\n# {0}", const 33 as i32, options(nostack, att_syntax));
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn instrument_atomic_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_atomic_read_write!(v, size);
instrument_atomic_check_alignment(v, size);
}

#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
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
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
}

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
#[inline(always)]
fn get_order(mut size: u64) -> i32 {
if unsafe { __slate_intrinsic_f217c4720f04f6db(size) } {
if size == 0 {
return 52i32;
}
if size < 4096u64 {
return 0;
}
return ilog2__2!(size.wrapping_sub(1)).wrapping_sub(12).wrapping_add(1);
}
size = size.wrapping_sub(1);
size >>= 12i32;
fls64(size)
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits123U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
}


#[inline(always)]
fn ERR_PTR(error: i64) -> *mut () {
error as *mut ()
}

#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
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
    __slate_bits_0: __slate_bits::__SlateBits128U0,
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
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
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

#[inline(always)]
fn arch_atomic_add_return(i: i32, mut v: *mut __SlateRecord_types_h_190) -> i32 {
let mut __v0: i32;
let mut __ret: i32 = i;
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
i.wrapping_add(__v0)
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
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
}

#[inline(always)]
fn raw_atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
arch_atomic_set(v, i);
}

#[inline(always)]
fn raw_atomic_add_return(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_add_return(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[inline(always)]
fn raw_atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_add(i, v)
}

#[inline(always)]
fn raw_atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_fetch_sub!(i, v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn raw_atomic_inc(v: *mut __SlateRecord_types_h_190) {
arch_atomic_inc(v);
}

#[inline(always)]
fn raw_atomic_inc_return(v: *mut __SlateRecord_types_h_190) -> i32 {
raw_atomic_add_return(1i32, v)
}

#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
raw_atomic64_read(v)
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
fn atomic_fetch_add_relaxed(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_add_relaxed(i, v)
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
fn atomic_fetch_sub_release(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_release!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_fetch_sub_release(i, v)
}

#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
}

#[inline(always)]
fn atomic_inc_return(v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc_return(v)
}

#[inline(always)]
fn atomic_long_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_read(v)
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


#[inline(always)]
fn might_resched() {
let mut __v0: Option<unsafe extern "C" fn() -> i32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn() -> i32>>(__SCT__might_resched as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__might_resched_42: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__might_resched) }) as *mut ();

fn might_fault() {
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
    __slate_bits_0: __slate_bits::__SlateBits129U0,
    pkru: u32,
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


// We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
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
struct workqueue_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
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
struct __call_single_node {
    llist: llist_node,
    __slate_anon_1: __SlateRecord_smp_types_h_60,
    src: u16,
    dst: u16,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct audit_context {
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
struct sysv_sem {
    undo_list: *mut sem_undo_list,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct refcount_struct {
    refs: __SlateRecord_types_h_190,
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
union __sifields {
    _kill: __SlateRecord_siginfo_h_39,
    _timer: __SlateRecord_siginfo_h_45,
    _rt: __SlateRecord_siginfo_h_53,
    _sigchld: __SlateRecord_siginfo_h_60,
    _sigfault: __SlateRecord_siginfo_h_69,
    _sigpoll: __SlateRecord_siginfo_h_103,
    _sigsys: __SlateRecord_siginfo_h_109,
}


// SIGPOLL (or any other signal without signal specific si_codes) si_codes
// data input available
// device disconnected
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
struct arch_tlbflush_unmap_batch {
    cpumask: cpumask,
    unmapped_pages: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
struct user_namespace {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_uidgid_types_h_13 {
    val: u32,
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
struct xol_area {
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
struct ldt_struct {
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
struct mem_cgroup {
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
    __slate_bits_0: __slate_bits::__SlateBits284U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
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
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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


// explicit shadow
// wait_event_interruptible - sleep until a condition gets true
// @wq_head: the waitqueue to wait on
// @condition: a C expression for the event to wait for
//
// The process is put to sleep (TASK_INTERRUPTIBLE) until the
// @condition evaluates to true or a signal is received.
// The @condition is checked each time the waitqueue @wq_head is woken up.
//
// wake_up() has to be called after changing any variable that could
// change the result of the wait condition.
//
// The function will return -ERESTARTSYS if it was interrupted by a
// signal and 0 if @condition evaluated to true.
fn __mutex_init(lock: *mut mutex, name: *const u8, key: *mut lock_class_key) {
unsafe { mutex_init_generic(lock) };
}

fn refcount_set(r: *mut refcount_struct, n: i32) {
atomic_set(unsafe { core::ptr::addr_of_mut!((*r).refs) }, n);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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

fn __refcount_inc(r: *mut refcount_struct, oldp: *mut i32) {
__refcount_add(1i32, r, oldp);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct folio_queue {
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


fn __refcount_dec_and_test(r: *mut refcount_struct, oldp: *mut i32) -> bool {
__refcount_sub_and_test(1i32, r, oldp)
}

fn refcount_dec_and_test(r: *mut refcount_struct) -> bool {
__refcount_dec_and_test(r, core::ptr::null_mut::<i32>())
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
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


fn kref_init(kref: *mut kref) {
refcount_set(unsafe { core::ptr::addr_of_mut!((*kref).refcount) }, 1i32);
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
// rcu_dereference() - fetch RCU-protected pointer for dereferencing
// @p: The pointer to read, prior to dereferencing
//
// This is a simple wrapper around rcu_dereference_check().
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_xarray_h_1175 {
    private_list: list_head,
    callback_head: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_xarray_h_1180 {
    tags: [[u64; 1]; 3],
    marks: [[u64; 1]; 3],
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


fn rwsem_assert_held_write_nolockdep(sem: *const rw_semaphore) {
let mut __v0: i64;
let __ret_warn_on: i32 = ((atomic_long_read(unsafe { core::ptr::addr_of!((*sem).count) }) as u64) & 1u64 == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("156: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 156b - ., 3; .popsection\n# {0}", const 156 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_rwsem_assert_held_write_nolockdep_1, sym __slate_asm_string_rwsem_assert_held_write_nolockdep_2, const 87 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("157: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 157b - ., 4; .popsection\n# {0}", const 157 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}

fn rwsem_assert_held_write(sem: *const rw_semaphore) {
if IS_ENABLED!() != 0 {
lockdep_assert_held_write!(sem);
} else {
rwsem_assert_held_write_nolockdep(sem);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
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
fn jiffies_to_msecs(j: u64) -> u32 {
1u64.wrapping_mul(j) as u32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


fn _msecs_to_jiffies(m: u32) -> u64 {
((m as i64).wrapping_add(1000 / 1000).wrapping_sub(1) / (1000 / 1000)) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct execute_work {
    work: work_struct,
}


fn __init_work(work: *mut work_struct, onstack: i32) {
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


fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
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


fn schedule_work(work: *mut work_struct) -> bool {
queue_work(unsafe { system_percpu_wq }, work)
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct backing_dev_info {
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
fn vma_flags_set_word(flags: *mut __SlateRecord_mm_types_h_871, value: u64) {
let bitmap: *mut u64 = unsafe { (*flags).__vma_flags.as_mut_ptr() as *mut u64 };
unsafe {
*bitmap = *bitmap | value;
}
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
fn compound_info_has_mask() -> bool {
// Limit mask usage to HugeTLB vmemmap optimization (HVO) where it
// makes a difference.
//
// The approach with mask would work in the wider set of conditions,
// but it requires validating that struct pages are naturally aligned
// for all orders up to the MAX_FOLIO_ORDER, which can be tricky.
if IS_ENABLED__2!() == 0 {
return false;
}
is_power_of_2(core::mem::size_of::<page>() as u64)
}

#[inline(always)]
fn _compound_head(page: *const page) -> u64 {
let mut info: u64 = 0;
let mut __v0: u64;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_187() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*page).__slate_anon_1.__slate_anon_2.compound_info) as *const u64) };
info = __v0;
let mut mask: u64 = 0;
if !compound_info_has_mask() {
// Bit 0 encodes PageTail()
if info & 1 != 0 {
return info.wrapping_sub(1);
}
return page as u64;
}
// If compound_info_has_mask() is true the rest of the info encodes
// the mask that converts the address of the tail page to the head page.
//
// No need to clear bit 0 in the mask as 'page' always has it clear.
//
// Let's do it in a branchless manner.
// Non-tail: -1UL, Tail: 0
mask = (info & 1).wrapping_sub(1);
// Non-tail: -1UL, Tail: info
mask |= info;
(page as u64) & mask
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
fn folio_test_slab(folio: *const folio) -> bool {
data_race!((unsafe { (*folio).__slate_anon_0.page.__slate_anon_2.page_type }) >> 24i32) == 245
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
fn folio_test_large_kmalloc(folio: *const folio) -> bool {
data_race!((unsafe { (*folio).__slate_anon_0.page.__slate_anon_2.page_type }) >> 24i32) == 248
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_lock {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
}


fn local_lock_release(l: *mut local_lock) {
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
struct fs_context {
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
core::arch::asm!("203: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 203b - ., 3; .popsection\n# {0}", const 203 as i32, options(nostack, att_syntax));
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
core::arch::asm!("204: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 204b - ., 4; .popsection\n# {0}", const 204 as i32, options(nostack, att_syntax));
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


// == WRITE
// == READ
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
struct fiemap_extent_info {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct xa_node {
    shift: u8,
    offset: u8,
    count: u8,
    nr_values: u8,
    parent: *mut xa_node,
    array: *mut xarray,
    __slate_anon_6: __SlateRecord_xarray_h_1175,
    slots: [*mut (); 64],
    __slate_anon_8: __SlateRecord_xarray_h_1180,
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
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
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct __SlateRecord_uuid_h_21 {
    b: [u8; 16],
}


// The length of a UUID string ("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
// not including trailing NUL.
fn mmap_assert_write_locked(mm: *const mm_struct) {
rwsem_assert_held_write(unsafe { core::ptr::addr_of!((*mm).__slate_anon_0.mmap_lock) });
}

// VMA locks do not behave like most ordinary locks found in the kernel, so we
// cannot quite have full lockdep tracking in the way we would ideally prefer.
//
// Read locks act as shared locks which exclude an exclusive lock being
// taken. We therefore mark these accordingly on read lock acquire/release.
//
// Write locks are acquired exclusively per-VMA, but released in a shared
// fashion, that is upon vma_end_write_all(), we update the mmap's seqcount such
// that write lock is released.
//
// We therefore cannot track write locks per-VMA, nor do we try. Mitigating this
// is the fact that, of course, we do lockdep-track the mmap lock rwsem which
// must be held when taking a VMA write lock.
//
// We do, however, want to indicate that during either acquisition of a VMA
// write lock or detachment of a VMA that we require the lock held be exclusive,
// so we utilise lockdep to do so.
fn __vma_raw_mm_seqnum(vma: *mut vm_area_struct) -> u32 {
let mm: *const mm_struct = (unsafe { (*vma).vm_mm }) as *const mm_struct;
// We must hold an exclusive write lock for this access to be valid.
mmap_assert_write_locked((unsafe { (*vma).vm_mm }) as *const mm_struct);
unsafe { (*mm).__slate_anon_0.mm_lock_seq.sequence }
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


fn __is_vma_write_locked(vma: *mut vm_area_struct) -> bool {
// current task is holding mmap_write_lock, both vma->vm_lock_seq and
// mm->mm_lock_seq can't be concurrently modified.
(unsafe { (*vma).vm_lock_seq }) == __vma_raw_mm_seqnum(vma)
}

fn vma_start_write(vma: *mut vm_area_struct) {
if __is_vma_write_locked(vma) {
return;
}
unsafe { __vma_start_write(vma, 2i32) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
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
struct assoc_array {
    root: *mut assoc_array_ptr,
    nr_leaves_on_tree: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
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
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
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


fn task_tgid_vnr(tsk: *mut task_struct) -> i32 {
unsafe { __task_pid_nr_ns(tsk, 1u32, core::ptr::null_mut::<pid_namespace>()) }
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
    __slate_bits_0: __slate_bits::__SlateBits287U0,
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
unsafe { __compiletime_assert_361() };
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
struct __SlateRecord_projid_h_24 {
    val: u32,
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct disk_stats {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct radix_tree_preload {
    lock: local_lock,
    nr: u32,
    nodes: *mut xa_node,
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct delayed_call {
    r#fn: Option<unsafe extern "C" fn(*mut ())>,
    arg: *mut (),
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
struct bio_alloc_cache {
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
    __slate_bits_0: __slate_bits::__SlateBits589U0,
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
struct elevator_queue {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
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
fn _kmalloc_array_noprof(n: u64, size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
let mut bytes: u64 = 0;
if unlikely!(check_mul_overflow!(n, size, core::ptr::addr_of_mut!(bytes))) != 0 {
return core::ptr::null_mut::<()>();
}
_kmalloc_noprof(bytes, flags, token)
}

// kcalloc - allocate memory for an array. The memory is set to zero.
// @n: number of elements.
// @size: element size.
// @flags: the type of memory to allocate (see kmalloc).
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
struct blk_mq_ctx {
}


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


// Get bit address from inode->i_state to use with wait_var_event()
// infrastructre.
fn iminor(inode: *const inode) -> u32 {
MINOR!(unsafe { (*inode).i_rdev })
}

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
struct bsg_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct offset_ctx {
    mt: maple_tree,
    next_offset: u64,
}


fn in_x32_syscall() -> bool {
false
}

fn in_32bit_syscall() -> bool {
in_ia32_syscall!() || in_x32_syscall()
}

fn in_compat_syscall() -> bool {
in_32bit_syscall()
}

// override the generic impl
#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_device_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_robust_list {
    next: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_transport_template {
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
struct proc_dir_entry {
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
struct idr {
    idr_rt: xarray,
    idr_base: u32,
    idr_next: u32,
}


// The IDR API does not expose the tagging functionality of the radix tree
// to users.  Use tag 0 to track whether a node has free space below it.
// Set the IDR flag and the IDR_FREE tag
// IDR_INIT() - Initialise an IDR.
// @name: Name of IDR.
//
// A freshly-initialised IDR contains no IDs.
// DEFINE_IDR() - Define a statically-allocated IDR.
// @name: Name of IDR.
//
// An IDR defined using this macro is ready for use with no additional
// initialisation required.  It contains no IDs.
fn idr_preload_end() {
loop {
__release__2!();
loop {
let mut __v1: *mut local_lock;
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<local_lock>() as *const ();
if true {
break;
}
}
let mut __v3: *mut local_lock;
let mut tcp_ptr__: u64 = 0;
let mut __v4: u64;
let mut pfo_val__: u64 = 0;
unsafe {
let __t0: u64;
core::arch::asm!("movq %gs:{1}(%rip), {0:r}", lateout(reg) __t0, sym this_cpu_off, options(pure, readonly, nostack, att_syntax));
pfo_val__ = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
__v4 = pfo_val__;
tcp_ptr__ = __v4;
let __v6: u64 = tcp_ptr__.wrapping_add((unsafe { core::ptr::addr_of_mut!(radix_tree_preloads.lock) }) as u64);
tcp_ptr__ = __v6;
__v3 = tcp_ptr__ as *mut local_lock;
__v1 = __v3;
let __l: *mut local_lock = __v1;
local_lock_release(__l);
if true {
break;
}
}
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
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_429: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

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
    __slate_bits_0: __slate_bits::__SlateBits661U0,
    vm_ops: *const vm_operations_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1624 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1624_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1624_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1624_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1624_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1624_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1625 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1625_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1625_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1625_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1625_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1625_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1651 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1651_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1651_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1651_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1651_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sg_c_1651_6 {
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
    pub struct __SlateBits284U0 {
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
    pub struct __SlateBits287U0 {
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
    pub struct __SlateBits123U0 {
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
    pub struct __SlateBits129U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits589U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits718U0 {
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
    pub struct __SlateBits715U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits718U1 {
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
    pub struct __SlateBits718U2 {
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
    pub struct __SlateBits718U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits661U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits601U0 {
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
    pub struct __SlateBits128U0 {
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
    pub struct __SlateBits843U0 {
        #[bits(1)]
        pub create: u32,
        #[bits(1)]
        pub single_lun: u32,
        #[bits(1)]
        pub pdt_1f_for_no_lun: u32,
        #[bits(1)]
        pub no_report_luns: u32,
        #[bits(1)]
        pub expecting_lun_change: u32,
        #[bits(3, access = na)]
        pub __slate_pad_5: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits850U0 {
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
    pub struct __SlateBits842U0 {
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
    #[bitfields::bitfield([u8; 8], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits841U0 {
        #[bits(1)]
        pub manage_system_start_stop: u32,
        #[bits(1)]
        pub manage_runtime_start_stop: u32,
        #[bits(1)]
        pub manage_shutdown: u32,
        #[bits(1)]
        pub manage_restart: u32,
        #[bits(1)]
        pub force_runtime_start_on_system_start: u32,
        #[bits(1)]
        pub is_ata: u32,
        #[bits(1)]
        pub removable: u32,
        #[bits(1)]
        pub changed: u32,
        #[bits(1)]
        pub busy: u32,
        #[bits(1)]
        pub lockable: u32,
        #[bits(1)]
        pub locked: u32,
        #[bits(1)]
        pub borken: u32,
        #[bits(1)]
        pub disconnect: u32,
        #[bits(1)]
        pub soft_reset: u32,
        #[bits(1)]
        pub sdtr: u32,
        #[bits(1)]
        pub wdtr: u32,
        #[bits(1)]
        pub ppr: u32,
        #[bits(1)]
        pub tagged_supported: u32,
        #[bits(1)]
        pub simple_tags: u32,
        #[bits(1)]
        pub was_reset: u32,
        #[bits(1)]
        pub expecting_cc_ua: u32,
        #[bits(1)]
        pub use_10_for_rw: u32,
        #[bits(1)]
        pub use_10_for_ms: u32,
        #[bits(1)]
        pub set_dbd_for_ms: u32,
        #[bits(1)]
        pub read_before_ms: u32,
        #[bits(1)]
        pub no_report_opcodes: u32,
        #[bits(1)]
        pub no_write_same: u32,
        #[bits(1)]
        pub use_16_for_rw: u32,
        #[bits(1)]
        pub use_16_for_sync: u32,
        #[bits(1)]
        pub skip_ms_page_8: u32,
        #[bits(1)]
        pub skip_ms_page_3f: u32,
        #[bits(1)]
        pub skip_vpd_pages: u32,
        #[bits(1)]
        pub try_vpd_pages: u32,
        #[bits(1)]
        pub use_192_bytes_for_3f: u32,
        #[bits(1)]
        pub no_start_on_add: u32,
        #[bits(1)]
        pub allow_restart: u32,
        #[bits(1)]
        pub start_stop_pwr_cond: u32,
        #[bits(1)]
        pub no_uld_attach: u32,
        #[bits(1)]
        pub select_no_atn: u32,
        #[bits(1)]
        pub fix_capacity: u32,
        #[bits(1)]
        pub guess_capacity: u32,
        #[bits(1)]
        pub retry_hwerror: u32,
        #[bits(1)]
        pub last_sector_bug: u32,
        #[bits(1)]
        pub no_read_disc_info: u32,
        #[bits(1)]
        pub no_read_capacity_16: u32,
        #[bits(1)]
        pub try_rc_10_first: u32,
        #[bits(1)]
        pub security_supported: u32,
        #[bits(1)]
        pub is_visible: u32,
        #[bits(1)]
        pub wce_default_on: u32,
        #[bits(1)]
        pub no_dif: u32,
        #[bits(1)]
        pub broken_fua: u32,
        #[bits(1)]
        pub lun_in_cdb: u32,
        #[bits(1)]
        pub unmap_limit_for_ws: u32,
        #[bits(1)]
        pub rpm_autosuspend: u32,
        #[bits(1)]
        pub ignore_media_change: u32,
        #[bits(1)]
        pub silence_suspend: u32,
        #[bits(1)]
        pub no_vpd_size: u32,
        #[bits(1)]
        pub cdl_supported: u32,
        #[bits(1)]
        pub cdl_enable: u32,
        #[bits(5, access = na)]
        pub __slate_pad_59: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits859U0 {
        #[bits(1)]
        pub twelve_byte: u32,
        #[bits(5)]
        pub target_status: u32,
        #[bits(8)]
        pub host_status: u32,
        #[bits(8)]
        pub driver_status: u32,
        #[bits(10)]
        pub other_flags: u32,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits601U0,
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


// One for each parameter, describing how to use it.  Some files do
// multiple of these per line, so can't just use MODULE_INFO.
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
struct module_version_attribute {
    mattr: module_attribute,
    module_name: *const u8,
    version: *const u8,
}


// module_init() - driver initialization entry point
// @x: function to be run at kernel boot time or module insertion
//
// module_init() will either be called during do_initcalls() (if
// builtin) or at module insertion time (if a module).  There can only
// be one per module.
// module_exit() - driver exit entry point
// @x: function to be run when driver is removed
//
// module_exit() will wrap the driver clean-up code
// with cleanup_module() when used with rmmod when
// the driver is a module.  If the driver is statically
// compiled into the kernel, module_exit() has no effect.
// There can only be one per module.
// For userspace: you can also call me...
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
// Author(s), use "Name <email>" or just "Name", for multiple
// authors use multiple MODULE_AUTHOR() statements/lines.
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


fn __page_ref_mod(page: *mut page, v: i32) {
}

fn page_ref_count(page: *const page) -> i32 {
atomic_read(unsafe { core::ptr::addr_of!((*page)._refcount) })
}

fn folio_ref_count(folio: &folio) -> i32 {
page_ref_count(unsafe { core::ptr::addr_of!((*(folio as *const folio)).__slate_anon_0.page) })
}

fn page_ref_inc(page: *mut page) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*page)._refcount) });
if page_ref_tracepoint_active!() != 0 {
__page_ref_mod(page, 1i32);
}
}

fn folio_ref_inc(folio: *mut folio) {
page_ref_inc(unsafe { core::ptr::addr_of_mut!((*folio).__slate_anon_0.page) });
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


fn vm_flags_set(vma: *mut vm_area_struct, flags: u64) {
vma_start_write(vma);
vma_flags_set_word(unsafe { core::ptr::addr_of_mut!((*vma).__slate_anon_3.flags) }, flags);
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
fn folio_get(folio: *mut folio) {
VM_BUG_ON_FOLIO!();
folio_ref_inc(folio);
}

fn get_page(page: *mut page) {
let folio: *mut folio = page_folio!(page);
let mut __v0: i64;
{
let __ret_warn_on: i32 = !(!folio_test_slab(folio as *const folio)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("469: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 469b - ., 3; .popsection\n# {0}", const 469 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_get_page_1, sym __slate_asm_string_get_page_2, const 2128 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("470: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 470b - ., 4; .popsection\n# {0}", const 470 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 != 0 {
return;
}
let mut __v5: i64;
let __ret_warn_on: i32 = !(!folio_test_large_kmalloc(folio as *const folio)) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("471: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 471b - ., 3; .popsection\n# {0}", const 471 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_get_page_6, sym __slate_asm_string_get_page_7, const 2130 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return;
}
folio_get(folio);
}

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
    __slate_bits_0: __slate_bits::__SlateBits718U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits718U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits718U2,
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
    __slate_bits_3: __slate_bits::__SlateBits718U3,
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
struct class_interface {
    node: list_head,
    class: *const class,
    add_dev: Option<unsafe extern "C" fn(*mut device) -> i32>,
    remove_dev: Option<unsafe extern "C" fn(*mut device)>,
}


core::arch::global_asm!(".section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_sg__623_2625_init_sg6:\t\t\t\n.long\t{0} - .\t\n.previous\t\t\t\t\t\n", sym init_sg, options(att_syntax));

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
    __slate_bits_0: __slate_bits::__SlateBits715U0,
}


fn dev_get_drvdata(dev: *const device) -> *mut () {
unsafe { (*dev).driver_data }
}

fn dev_set_drvdata(dev: *mut device, data: *mut ()) {
unsafe {
(*dev).driver_data = data;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cdev {
    kobj: kobject,
    owner: *mut module,
    ops: *const file_operations,
    list: list_head,
    dev: u32,
    count: u32,
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


#[inline(always)]
fn seq_puts(m: *mut seq_file, s: *const u8) {
if !unsafe { __slate_intrinsic_70b71f185ebb8c4a(*s) } {
unsafe { __seq_puts(m, s) };
} else {
if (unsafe { *s.wrapping_offset(0 as isize) }) != 0 && (unsafe { *s.wrapping_offset(1 as isize) }) == 0 {
unsafe { seq_putc(m, *s.wrapping_offset(0 as isize)) };
} else {
unsafe { seq_write(m, s as *const (), __builtin_strlen(s)) };
}
}
}

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


fn queue_max_sectors(q: *const request_queue) -> u32 {
unsafe { (*q).limits.max_sectors }
}

fn queue_max_segments(q: *const request_queue) -> u16 {
unsafe { (*q).limits.max_segments }
}

fn blk_lim_dma_alignment_and_pad(lim: *mut queue_limits) -> u32 {
(unsafe { (*lim).dma_alignment }) | unsafe { (*lim).dma_pad_mask }
}

fn blk_rq_aligned(q: *mut request_queue, addr: u64, len: u32) -> bool {
let alignment: u32 = blk_lim_dma_alignment_and_pad(unsafe { core::ptr::addr_of_mut!((*q).limits) });
addr & (alignment as u64) == 0 && len & alignment == 0
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


// These macros should be used after a dma_map_sg call has been done
// to get bus addresses of each of the SG entries and their lengths.
// You should only work with the number of sg entries dma_map_sg
// returns, or alternatively stop on the first sg_dma_len(sg) which
// is 0.
#[repr(C)]
#[derive(Clone, Copy)]
struct sg_table {
    sgl: *mut scatterlist,
    nents: u32,
    orig_nents: u32,
}


// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
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


fn blk_mq_rq_to_pdu(rq: *mut request) -> *mut () {
rq.wrapping_offset(1 as isize) as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_map_data {
    pages: *mut *mut page,
    offset: u64,
    page_order: u16,
    nr_entries: u16,
    null_mapped: bool,
    from_user: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rchan_buf_stats {
    full_count: u32,
    big_count: u32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct rchan_buf {
    start: *mut (),
    data: *mut (),
    offset: u64,
    subbufs_produced: u64,
    subbufs_consumed: u64,
    chan: *mut rchan,
    read_wait: wait_queue_head,
    wakeup_work: irq_work,
    dentry: *mut dentry,
    kref: kref,
    stats: rchan_buf_stats,
    page_array: *mut *mut page,
    page_count: u32,
    finalized: u32,
    padding: *mut u64,
    bytes_consumed: u64,
    early_bytes: u64,
    cpu: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rchan {
    version: u32,
    subbuf_size: u64,
    n_subbufs: u64,
    alloc_size: u64,
    cb: *const rchan_callbacks,
    kref: kref,
    private_data: *mut (),
    buf: *mut *mut rchan_buf,
    is_global: i32,
    list: list_head,
    parent: *mut dentry,
    has_base_filename: i32,
    base_filename: [u8; 255],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rchan_callbacks {
    subbuf_start: Option<unsafe extern "C" fn(*mut rchan_buf, *mut (), *mut ()) -> i32>,
    create_buf_file: Option<unsafe extern "C" fn(*const u8, *mut dentry, u16, *mut rchan_buf, *mut i32) -> *mut dentry>,
    remove_buf_file: Option<unsafe extern "C" fn(*mut dentry) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blk_trace {
    version: i32,
    trace_state: i32,
    rchan: *mut rchan,
    sequence: *mut u64,
    msg_data: *mut u8,
    act_mask: u64,
    start_lba: u64,
    end_lba: u64,
    pid: u32,
    dev: u32,
    dir: *mut dentry,
    running_list: list_head,
    dropped: __SlateRecord_types_h_190,
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
struct scsi_vpd {
    rcu: callback_head,
    len: i32,
    data: [u8; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_device {
    host: *mut Scsi_Host,
    request_queue: *mut request_queue,
    siblings: list_head,
    same_target_siblings: list_head,
    budget_map: sbitmap,
    device_blocked: __SlateRecord_types_h_190,
    restarts: __SlateRecord_types_h_190,
    list_lock: spinlock,
    starved_entry: list_head,
    queue_depth: u16,
    max_queue_depth: u16,
    last_queue_full_depth: u16,
    last_queue_full_count: u16,
    last_queue_full_time: u64,
    queue_ramp_up_period: u64,
    last_queue_ramp_up: u64,
    id: u32,
    channel: u32,
    lun: u64,
    manufacturer: u32,
    sector_size: u32,
    hostdata: *mut (),
    r#type: u8,
    scsi_level: u8,
    inq_periph_qual: u8,
    inquiry_mutex: mutex,
    inquiry_len: u8,
    inquiry: *mut u8,
    vendor: [u8; 9],
    model: [u8; 17],
    rev: [u8; 5],
    vpd_pg0: *mut scsi_vpd,
    vpd_pg83: *mut scsi_vpd,
    vpd_pg80: *mut scsi_vpd,
    vpd_pg89: *mut scsi_vpd,
    vpd_pgb0: *mut scsi_vpd,
    vpd_pgb1: *mut scsi_vpd,
    vpd_pgb2: *mut scsi_vpd,
    vpd_pgb7: *mut scsi_vpd,
    sdev_target: *mut scsi_target,
    sdev_bflags: u64,
    eh_timeout: u32,
    __slate_bits_0: __slate_bits::__SlateBits841U0,
    queue_stopped: u32,
    offline_already: bool,
    ua_new_media_ctr: __SlateRecord_types_h_190,
    ua_por_ctr: __SlateRecord_types_h_190,
    disk_events_disable_depth: __SlateRecord_types_h_190,
    supported_events: [u64; 1],
    pending_events: [u64; 1],
    event_list: list_head,
    event_work: work_struct,
    max_device_blocked: u32,
    iorequest_cnt: __SlateRecord_types_h_190,
    iodone_cnt: __SlateRecord_types_h_190,
    ioerr_cnt: __SlateRecord_types_h_190,
    iotmo_cnt: __SlateRecord_types_h_190,
    sdev_gendev: device,
    sdev_dev: device,
    requeue_work: work_struct,
    handler: *mut scsi_device_handler,
    handler_data: *mut (),
    dma_drain_len: u64,
    dma_drain_buf: *mut (),
    sg_timeout: u32,
    sg_reserved_size: u32,
    bsg_dev: *mut bsg_device,
    access_state: u8,
    state_mutex: mutex,
    sdev_state: u32,
    quiesced_by: *mut task_struct,
    sdev_data: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_target {
    starget_sdev_user: *mut scsi_device,
    siblings: list_head,
    devices: list_head,
    dev: device,
    reap_ref: kref,
    channel: u32,
    id: u32,
    __slate_bits_0: __slate_bits::__SlateBits843U0,
    target_busy: __SlateRecord_types_h_190,
    target_blocked: __SlateRecord_types_h_190,
    can_queue: u32,
    max_target_blocked: u32,
    scsi_level: u8,
    state: u32,
    hostdata: *mut (),
    starget_data: [u64; 0],
}


fn scsi_target(sdev: *mut scsi_device) -> *mut scsi_target {
to_scsi_target!(unsafe { (*sdev).sdev_gendev.parent })
}

fn scsi_device_online(sdev: *mut scsi_device) -> i32 {
(((unsafe { (*sdev).sdev_state }) as u32) != 6 && ((unsafe { (*sdev).sdev_state }) as u32) != 7 && ((unsafe { (*sdev).sdev_state }) as u32) != 4) as i32
}

fn scsi_device_busy(sdev: *mut scsi_device) -> i32 {
(unsafe { sbitmap_weight(core::ptr::addr_of_mut!((*sdev).budget_map) as *const sbitmap) }) as i32
}

#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_data_buffer {
    table: sg_table,
    length: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_cmnd {
    device: *mut scsi_device,
    eh_entry: list_head,
    abort_work: delayed_work,
    rcu: callback_head,
    eh_eflags: i32,
    budget_token: i32,
    jiffies_at_alloc: u64,
    retries: i32,
    allowed: i32,
    prot_op: u8,
    prot_type: u8,
    prot_flags: u8,
    submitter: u8,
    cmd_len: u16,
    sc_data_direction: u32,
    cmnd: [u8; 32],
    sdb: scsi_data_buffer,
    prot_sdb: *mut scsi_data_buffer,
    underflow: u32,
    transfersize: u32,
    resid_len: u32,
    sense_len: u32,
    sense_buffer: *mut u8,
    flags: i32,
    state: u64,
    extra_len: u32,
    host_scribble: *mut u8,
    result: i32,
}


fn scsi_sense_is_deferred(sshdr: &scsi_sense_hdr) -> bool {
((unsafe { (*(sshdr as *const scsi_sense_hdr)).response_code }) as i32) >= 112 && ((unsafe { (*(sshdr as *const scsi_sense_hdr)).response_code }) as i32) & 1 != 0
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
    __slate_bits_0: __slate_bits::__SlateBits850U0,
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
    __slate_bits_0: __slate_bits::__SlateBits842U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct sg_io_hdr {
    interface_id: i32,
    dxfer_direction: i32,
    cmd_len: u8,
    mx_sb_len: u8,
    iovec_count: u16,
    dxfer_len: u32,
    dxferp: *mut (),
    cmdp: *mut u8,
    sbp: *mut (),
    timeout: u32,
    flags: u32,
    pack_id: i32,
    usr_ptr: *mut (),
    status: u8,
    masked_status: u8,
    msg_status: u8,
    sb_len_wr: u8,
    host_status: u16,
    driver_status: u16,
    resid: i32,
    duration: u32,
    info: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct compat_sg_io_hdr {
    interface_id: i32,
    dxfer_direction: i32,
    cmd_len: u8,
    mx_sb_len: u8,
    iovec_count: u16,
    dxfer_len: u32,
    dxferp: u32,
    cmdp: u32,
    sbp: u32,
    timeout: u32,
    flags: u32,
    pack_id: i32,
    usr_ptr: u32,
    status: u8,
    masked_status: u8,
    msg_status: u8,
    sb_len_wr: u8,
    host_status: u16,
    driver_status: u16,
    resid: i32,
    duration: u32,
    info: u32,
}


// Use negative values to flag difference from original sg_header structure
// e.g. a SCSI Test Unit Ready command
// e.g. a SCSI WRITE command
// e.g. a SCSI READ command
// treated like SG_DXFER_FROM_DEV with the
// additional property than during indirect
// IO the user buffer is copied into the
// kernel buffers before the transfer
// Unknown data direction
// following flag values can be "or"-ed together
// default is indirect IO
// command block (when <= SCSI_2)
// request memory mapped IO
// user space (debug indirect IO)
//
// defaults:: for sg driver: Q_AT_HEAD; for block layer: Q_AT_TAIL
// something abnormal happened
// direct IO requested and performed
// Obsolete DRIVER_SENSE driver byte
//
// Originally the SCSI midlayer would set the DRIVER_SENSE driver byte when
// a sense code was generated and a sense buffer was allocated.
// However, as nowadays every scsi command has a sense code allocated this
// distinction became moot as one could check the sense buffer directly.
// Consequently this byte is not set anymore from the midlayer, but SG will
// keep setting this byte to be compatible with previous releases.
#[repr(C)]
#[derive(Clone, Copy)]
struct sg_scsi_id {
    host_no: i32,
    channel: i32,
    scsi_id: i32,
    lun: i32,
    scsi_type: i32,
    h_cmd_per_lun: i16,
    d_queue_depth: i16,
    unused: [i32; 2],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sg_req_info {
    req_state: u8,
    orphan: u8,
    sg_io_owned: u8,
    problem: u8,
    pack_id: i32,
    usr_ptr: *mut (),
    duration: u32,
    unused: i32,
}


// true for emulated host adapter (ATAPI)
// 3rd arg = 0 to disable transform, 1 to enable it
// request a new reserved buffer size
// actual size of reserved buffer
// The following ioctl has a 'sg_scsi_id_t *' object as its 3rd argument.
// Yields fd's bus, chan, dev, lun + type
// Override host setting and always DMA using low memory ( <16MB on i386)
// 0-> use adapter setting, 1-> force
// 0-> use all ram for dma; 1-> low dma ram
// When SG_SET_FORCE_PACK_ID set to 1, pack_id is input to read() which
// tries to fetch a packet with a matching pack_id, waits, or returns EAGAIN.
// If pack_id is -1 then read oldest waiting. When ...FORCE_PACK_ID set to 0
// then pack_id ignored by read() and oldest readable fetched.
// Yields oldest readable pack_id (or -1)
// Number of commands awaiting read()
// Yields max scatter gather tablesize allowed by current host adapter
// 0 implies can't do scatter gather
// Example: version 2.1.34 yields 20134
// Returns -EBUSY if occupied. 3rd argument pointer to int (see next)
// synchronous SCSI command ioctl, (only in version 3 interface)
// similar effect as write() followed by read()
// yields table of active requests
// How to treat EINTR during SG_IO ioctl(), only in SG 3.x series
// 1 -> hold for read(), 0 -> drop (def)
// yields scsi midlevel's access_count for this SCSI device
// Defaults, commented if they differ from original sg driver
// load time option
// maximum outstanding requests, write() yields EDOM if exceeded
// this only applies to the sg_header interface
#[repr(C)]
#[derive(Clone, Copy)]
struct sg_header {
    pack_len: i32,
    reply_len: i32,
    pack_id: i32,
    result: i32,
    __slate_bits_0: __slate_bits::__SlateBits859U0,
    sense_buffer: [u8; 16],
}


// unit: jiffies (10ms on i386)
// yield timeout as _return_ value
// Get/set command queuing state per fd (default is SG_DEF_COMMAND_Q.
// Each time a sg_io_hdr_t object is seen on this file descriptor, this
// command queuing flag is set on (overriding the previous setting).
// Yields 0 (queuing off) or 1 (on)
// Change queuing state with 0 or 1
// Turn on/off error sense trace (1 and 0 respectively, default is off).
// Try using: "# cat /proc/scsi/sg/debug" instead in the v3 driver
// 0 -> turn off debug
// override SCSI command length with given
// number on the next write() on this file descriptor
// HZ == 'jiffies in 1 second'
// command queuing is always on when
// the new interface is used
#[repr(C)]
#[derive(Clone, Copy)]
struct proc_ops {
    proc_flags: u32,
    proc_open: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    proc_read: Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>,
    proc_read_iter: Option<unsafe extern "C" fn(*mut kiocb, *mut iov_iter) -> i64>,
    proc_write: Option<unsafe extern "C" fn(*mut file, *const u8, u64, *mut i64) -> i64>,
    proc_lseek: Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>,
    proc_release: Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>,
    proc_poll: Option<unsafe extern "C" fn(*mut file, *mut poll_table_struct) -> u32>,
    proc_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    proc_compat_ioctl: Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>,
    proc_mmap: Option<unsafe extern "C" fn(*mut file, *mut vm_area_struct) -> i32>,
    proc_get_unmapped_area: Option<unsafe extern "C" fn(*mut file, u64, u64, u64, u64) -> u64>,
}


// D. P. Gilbert (dgilbert@interlog.com), notes:
//     - scsi logging is available via SCSI_LOG_TIMEOUT macros. First
//       the kernel/module needs to be built with CONFIG_SCSI_LOGGING
//       (otherwise the macros compile to empty statements).
// for sg_check_file_access()
static mut sg_version_date: *mut u8 = b"20140603\0".as_ptr() as *mut u8;

// SG_MAX_CDB_SIZE should be 260 (spc4r37 section 3.1.30) however the type
// of sg_io_hdr::cmd_len can only represent 255. All SCSI commands greater
// than 16 bytes are "variable length" whose length is a multiple of 4
// N.B. This variable is readable and writeable via
// /proc/scsi/sg/def_reserved_size . Each time sg_open() is called a buffer
// of this size (or less if there is not enough memory) will be reserved
// for use by this file descriptor.
/// picks up init parameter
static mut def_reserved_size: i32 = 32768;

static mut sg_allow_dio: i32 = 0;

static mut scatter_elem_sz: i32 = 32768;

static mut scatter_elem_sz_prev: i32 = 32768;

static mut sg_index_idr: idr = idr { idr_rt: xarray { xa_lock: spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} }, xa_flags: 16777220u32, xa_head: core::ptr::null_mut::<()>() }, idr_base: 0u32, idr_next: 0u32 };

/// Also used to lock
/// file descriptor list for device
static mut sg_index_lock: rwlock = rwlock { raw_lock: qrwlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qrwlock_types_h_27 = unsafe { core::mem::zeroed() };
__t0.cnts = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
}, wait_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } } };

static mut sg_interface: class_interface = {
let mut __t0: class_interface = unsafe { core::mem::zeroed() };
__t0.add_dev = Some(sg_add_device);
__t0.remove_dev = Some(sg_remove_device);
__t0
};

#[repr(C)]
#[derive(Clone, Copy)]
struct sg_scatter_hold {
    /// Count of kernel scatter-gather pieces
    k_use_sg: u16,
    /// size of malloc'd scatter-gather list ++
    sglist_len: u32,
    /// Size of (aggregate) data buffer
    bufflen: u32,
    pages: *mut *mut page,
    page_order: i32,
    /// 0->indirect IO (or mmap), 1->dio
    dio_in_use: u8,
    /// first byte of command
    cmd_opcode: u8,
    // holding area for scsi scatter gather info
}


// forward declarations
#[repr(C)]
#[derive(Clone, Copy)]
struct sg_request {
    /// list entry
    entry: list_head,
    /// NULL -> not in use
    parentfp: *mut sg_fd,
    /// hold buffer, perhaps scatter list
    data: sg_scatter_hold,
    /// scsi command+info, see <scsi/sg.h>
    header: sg_io_hdr,
    sense_b: [u8; 96],
    /// 1 -> using reserve buffer, 0 -> not ...
    res_used: u8,
    /// 1 -> drop on sight, 0 -> normal
    orphan: u8,
    /// 1 -> packet belongs to SG_IO
    sg_io_owned: u8,
    /// done protected by rq_list_lock
    /// 0->before bh, 1->before read, 2->read
    done: u8,
    rq: *mut request,
    bio: *mut bio,
    ew: execute_work,
    // SG_MAX_QUEUE requests outstanding per file
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sg_fd {
    /// protected by device's sfd_lock
    sfd_siblings: list_head,
    /// owning device
    parentdp: *mut sg_device,
    /// queue read until command done
    read_wait: wait_queue_head,
    /// protect access to list in req_arr
    rq_list_lock: rwlock,
    /// protect against changes in this fd
    f_mutex: mutex,
    /// defaults to SG_DEFAULT_TIMEOUT
    timeout: i32,
    /// defaults to SG_DEFAULT_TIMEOUT_USER
    timeout_user: i32,
    /// buffer held for this file descriptor
    reserve: sg_scatter_hold,
    /// head of request list
    rq_list: list_head,
    /// used by asynchronous notification
    async_qp: *mut fasync_struct,
    /// used as singly-linked list
    req_arr: [sg_request; 16],
    /// 1 -> pack_id input to read(), 0 -> ignored
    force_packid: u8,
    /// 1 -> allow command queuing, 0 -> don't
    cmd_q: u8,
    /// 0: automatic, >0: use on next write()
    next_cmd_len: u8,
    /// 0 -> drop orphan (def), 1 -> keep for read()
    keep_orphan: u8,
    /// 0 -> mmap() never called on this fd
    mmap_called: u8,
    /// 1 -> 'reserve' array in use
    res_in_use: u8,
    f_ref: kref,
    ew: execute_work,
    // holds the state of a file descriptor
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sg_device {
    device: *mut scsi_device,
    /// queue open() when O_EXCL present
    open_wait: wait_queue_head,
    /// held when in open() or release()
    open_rel_lock: mutex,
    /// adapter's max scatter-gather table size
    sg_tablesize: i32,
    /// device index number
    index: u32,
    sfds: list_head,
    /// protect access to sfd list
    sfd_lock: rwlock,
    /// 0->device usable, 1->device detaching
    detaching: __SlateRecord_types_h_190,
    /// 1->open(O_EXCL) succeeded and is active
    exclude: bool,
    /// count of opens (perhaps < num(sfds) )
    open_cnt: i32,
    /// 0->off, 1->sense, 9->dump dev, 10-> all devs
    sgdebug: u8,
    name: [u8; 32],
    /// char_dev [sysfs: /sys/cdev/major/sg<n>]
    cdev: *mut cdev,
    d_ref: kref,
    // holds the state of each scsi generic device
}


/// The SCSI interfaces that use read() and write() as an asynchronous variant of
/// ioctl(..., SG_IO, ...) are fundamentally unsafe, since there are lots of ways
/// to trigger read() and write() calls from various contexts with elevated
/// privileges. This can lead to kernel memory corruption (e.g. if these
/// interfaces are called through splice()) and privilege escalation inside
/// userspace (e.g. if a process with access to such a device passes a file
/// descriptor to a SUID binary as stdin/stdout/stderr).
///
/// This function provides protection for the legacy API by restricting the
/// calling context.
fn sg_check_file_access(filp: *mut file, caller: *const u8) -> i32 {
if (unsafe { (*filp).f_cred }) != current_real_cred!() {
let mut __v2: i64;
let __ret_do_once: bool = true;
let mut __v3: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely!(__ret_cond) != 0 && unlikely!(!unsafe { __already_done }) != 0 {
unsafe {
__already_done = true;
}
__ret_once = true;
}
__v3 = __ret_once as i64;
if __v3 != 0 {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x013%s: process %d (%s) changed security contexts after opening file descriptor, this is not allowed.\n".as_ptr() as *const u8, caller, task_tgid_vnr(get_current()), (*get_current()).comm.as_mut_ptr() as *mut u8) };
}
__v2 = __ret_do_once as i64;
return 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".data..once")]
static mut __already_done: bool = unsafe { core::mem::zeroed() };

fn sg_allow_access(filp: *mut file, cmd: *mut u8) -> i32 {
let sfp: *mut sg_fd = (unsafe { (*filp).private_data }) as *mut sg_fd;
if ((unsafe { (*(*(*sfp).parentdp).device).r#type }) as i32) == 6 {
return 0;
}
if !unsafe { scsi_cmd_allowed(cmd, (*filp).f_mode & 2u32 != 0) } {
return 1i32.wrapping_neg();
}
0
}

fn open_wait(sdp: *mut sg_device, flags: i32) -> i32 {
let mut __int: i64 = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_1: i32 = unsafe { core::mem::zeroed() };
let mut __int_1: i64 = unsafe { core::mem::zeroed() };
let mut __ret_2: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_3: i32 = unsafe { core::mem::zeroed() };
let mut retval: i32 = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
retval = 0;
if flags & 128i32 != 0 {
loop {
if (*sdp).open_cnt > 0 {
mutex_unlock(core::ptr::addr_of_mut!((*sdp).open_rel_lock));
__ret_1 = 0;
might_sleep!();
if !(atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 || (*sdp).open_cnt == 0) {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = 0;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_2), if false { 1i32 } else { 0i32 });
'__join_6: {
'__join_7: {
loop {
__int = prepare_to_wait_event(core::ptr::addr_of_mut!((*sdp).open_wait), core::ptr::addr_of_mut!(*__slate_slot_2), 1i32);
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 || (*sdp).open_cnt == 0 {
break '__join_7;
} else {
if ___wait_is_interruptible!(1i32) && __int != 0 {
break;
} else {
schedule();
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 || (*sdp).open_cnt == 0 {
break '__join_7;
}
}
}
}
__ret = __int;
break '__join_6;
}
finish_wait(core::ptr::addr_of_mut!((*sdp).open_wait), core::ptr::addr_of_mut!(*__slate_slot_2));
}
__ret_1 = __ret as i32;
}
retval = __ret_1;
mutex_lock(core::ptr::addr_of_mut!((*sdp).open_rel_lock));
if retval != 0 {
return retval;
} else {
// -ERESTARTSYS
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 {
break;
}
}
} else {
break '__join_0;
}
}
return 19i32.wrapping_neg();
} else {
loop {
if (*sdp).exclude {
mutex_unlock(core::ptr::addr_of_mut!((*sdp).open_rel_lock));
__ret_3 = 0;
might_sleep!();
if !(atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 || !(*sdp).exclude) {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret_2 = 0;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_8), if false { 1i32 } else { 0i32 });
'__join_19: {
'__join_20: {
loop {
__int_1 = prepare_to_wait_event(core::ptr::addr_of_mut!((*sdp).open_wait), core::ptr::addr_of_mut!(*__slate_slot_8), 1i32);
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 || !(*sdp).exclude {
break '__join_20;
} else {
if ___wait_is_interruptible!(1i32) && __int_1 != 0 {
break;
} else {
schedule();
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 || !(*sdp).exclude {
break '__join_20;
}
}
}
}
__ret_2 = __int_1;
break '__join_19;
}
finish_wait(core::ptr::addr_of_mut!((*sdp).open_wait), core::ptr::addr_of_mut!(*__slate_slot_8));
}
__ret_3 = __ret_2 as i32;
}
retval = __ret_3;
mutex_lock(core::ptr::addr_of_mut!((*sdp).open_rel_lock));
if retval != 0 {
return retval;
} else {
// -ERESTARTSYS
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 {
break;
}
}
} else {
break '__join_0;
}
}
return 19i32.wrapping_neg();
}
}
return retval;
}
}

/// Returns 0 on success, else a negated errno value
#[unsafe(link_section = ".text.slate_distinct.sg.sg_open")]
extern "C" fn sg_open(inode: *mut inode, filp: *mut file) -> i32 {
// scsi_block_when_processing_errors() may block so bypass
// check if O_NONBLOCK. Permits SCSI commands to be issued
// during error recovery. Tread carefully.
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut sfp: core::mem::MaybeUninit<*mut sg_fd> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut sg_fd = core::ptr::addr_of_mut!(sfp) as *mut *mut sg_fd;
let mut sdp: core::mem::MaybeUninit<*mut sg_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut sg_device = core::ptr::addr_of_mut!(sdp) as *mut *mut sg_device;
let mut device: core::mem::MaybeUninit<*mut scsi_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut scsi_device = core::ptr::addr_of_mut!(device) as *mut *mut scsi_device;
let mut q: core::mem::MaybeUninit<*mut request_queue> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut request_queue = core::ptr::addr_of_mut!(q) as *mut *mut request_queue;
let mut flags: i32 = unsafe { core::mem::zeroed() };
let mut dev: i32 = unsafe { core::mem::zeroed() };
unsafe {
dev = iminor(inode as *const inode) as i32;
flags = (*filp).f_flags as i32;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
nonseekable_open(inode, filp);
if flags & 128i32 != 0 && 0 == flags & 3 {
return 1i32.wrapping_neg();
} else {
// Can't lock it with read only access
*__slate_slot_6 = sg_get_dev(dev);
if IS_ERR(*__slate_slot_6 as *const ()) {
return PTR_ERR(*__slate_slot_6 as *const ()) as i32;
} else {
'__join_3: {
SCSI_LOG_TIMEOUT!();
// This driver's module count bumped by fops_get in <linux/fs.h>
//
// Prevent the device driver from vanishing while we sleep
*__slate_slot_7 = (*(*__slate_slot_6)).device;
*__slate_slot_4 = scsi_device_get(*__slate_slot_7);
if *__slate_slot_4 == 0 {
if flags & 2048i32 != 0 {
__v3 = true;
} else {
__v3 = scsi_block_when_processing_errors(*__slate_slot_7) != 0;
}
if !__v3 {
*__slate_slot_4 = 6i32.wrapping_neg();
// we are in error recovery for this device
} else {
'__join_1: {
mutex_lock(core::ptr::addr_of_mut!((*(*__slate_slot_6)).open_rel_lock));
if flags & 2048i32 != 0 {
if flags & 128i32 != 0 {
if (*(*__slate_slot_6)).open_cnt > 0 {
*__slate_slot_4 = 16i32.wrapping_neg();
break '__join_1;
}
} else {
if (*(*__slate_slot_6)).exclude {
*__slate_slot_4 = 16i32.wrapping_neg();
break '__join_1;
}
}
} else {
*__slate_slot_4 = open_wait(*__slate_slot_6, flags);
if *__slate_slot_4 != 0 {
break '__join_1;
}
// -ERESTARTSYS or -ENODEV
}
// N.B. at this point we are holding the open_rel_lock
if flags & 128i32 != 0 {
(*(*__slate_slot_6)).exclude = true;
}
if (*(*__slate_slot_6)).open_cnt < 1 {
// no existing opens
(*(*__slate_slot_6)).sgdebug = 0;
*__slate_slot_8 = (*(*__slate_slot_7)).request_queue;
(*(*__slate_slot_6)).sg_tablesize = queue_max_segments(*__slate_slot_8 as *const request_queue) as i32;
}
*__slate_slot_5 = sg_add_sfp(*__slate_slot_6);
if IS_ERR(*__slate_slot_5 as *const ()) {
*__slate_slot_4 = PTR_ERR(*__slate_slot_5 as *const ()) as i32;
if flags & 128i32 != 0 {
(*(*__slate_slot_6)).exclude = false;
// undo if error
wake_up_interruptible!(core::ptr::addr_of_mut!((*(*__slate_slot_6)).open_wait));
}
} else {
(*filp).private_data = *__slate_slot_5 as *mut ();
let __v2: *mut sg_device = *__slate_slot_6;
let __v1: i32 = (*__v2).open_cnt;
(*__v2).open_cnt = __v1.wrapping_add(1);
mutex_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_6)).open_rel_lock));
*__slate_slot_4 = 0;
break '__join_3;
}
}
mutex_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_6)).open_rel_lock));
}
kref_put(core::ptr::addr_of_mut!((*(*__slate_slot_6)).d_ref), Some(sg_device_destroy));
scsi_device_put(*__slate_slot_7);
return *__slate_slot_4;
}
}
kref_put(core::ptr::addr_of_mut!((*(*__slate_slot_6)).d_ref), Some(sg_device_destroy));
return *__slate_slot_4;
}
}
}
}

/// Release resources associated with a successful sg_open()
/// Returns 0 on success, else a negated errno value
#[unsafe(link_section = ".text.slate_distinct.sg.sg_release")]
extern "C" fn sg_release(inode: *mut inode, filp: *mut file) -> i32 {
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let __v0: *mut sg_fd = (unsafe { (*filp).private_data }) as *mut sg_fd;
sfp = __v0;
let __v1: bool = if __v0.is_null() {
true
} else {
let __v2: *mut sg_device = unsafe { (*sfp).parentdp };
sdp = __v2;
__v2.is_null()
};
if __v1 {
return 6i32.wrapping_neg();
}
SCSI_LOG_TIMEOUT!();
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sdp).open_rel_lock)) };
let __v5: i32 = unsafe { (*sdp).open_cnt };
unsafe {
(*sdp).open_cnt = __v5.wrapping_sub(1);
}
// possibly many open()s waiting on exlude clearing, start many;
// only open(O_EXCL)s wait on 0==open_cnt so only start one
if unsafe { (*sdp).exclude } {
unsafe {
(*sdp).exclude = false;
}
wake_up_interruptible_all!(unsafe { core::ptr::addr_of_mut!((*sdp).open_wait) });
} else {
if 0 == unsafe { (*sdp).open_cnt } {
wake_up_interruptible!(unsafe { core::ptr::addr_of_mut!((*sdp).open_wait) });
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sdp).open_rel_lock)) };
kref_put(unsafe { core::ptr::addr_of_mut!((*sfp).f_ref) }, Some(sg_remove_sfp));
0
}

fn get_sg_io_pack_id(pack_id: &mut i32, buf: *mut (), count: u64) -> i32 {
let mut old_hdr: *mut sg_header = buf as *mut sg_header;
let mut reply_len: i32 = 0;
if count >= (core::mem::size_of::<sg_header>() as u64) {
// negative reply_len means v3 format, otherwise v1/v2
let mut __v0: i64;
might_fault();
let mut __v1: i64;
{
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*old_hdr).reply_len) as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
instrument_get_user!(__val_gu);
reply_len = __val_gu as i32;
__v1 = __ret_gu as i64;
}
__v0 = __v1;
if __v0 != 0 {
return 14i32.wrapping_neg();
}
if reply_len >= 0 {
let mut __v4: i64;
might_fault();
let mut __v5: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t2: u64;
let __t3: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*old_hdr).pack_id) as u64 => __t2, lateout("dx") __t3, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
instrument_get_user!(__val_gu);
unsafe {
*(pack_id as *mut i32) = __val_gu as i32;
}
__v5 = __ret_gu as i64;
__v4 = __v5;
return __v4 as i32;
}
if in_compat_syscall() && count >= (core::mem::size_of::<compat_sg_io_hdr>() as u64) {
let mut hp: *mut compat_sg_io_hdr = buf as *mut compat_sg_io_hdr;
let mut __v8: i64;
might_fault();
let mut __v9: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t4: u64;
let __t5: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*hp).pack_id) as u64 => __t4, lateout("dx") __t5, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t4 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t5 as u32) };
}
instrument_get_user!(__val_gu);
unsafe {
*(pack_id as *mut i32) = __val_gu as i32;
}
__v9 = __ret_gu as i64;
__v8 = __v9;
return __v8 as i32;
}
if count >= (core::mem::size_of::<sg_io_hdr>() as u64) {
let mut hp: *mut sg_io_hdr = buf as *mut sg_io_hdr;
let mut __v12: i64;
might_fault();
let mut __v13: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u32 = 0;
__chk_user_ptr!();
unsafe {
let __t6: u64;
let __t7: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") core::ptr::addr_of_mut!((*hp).pack_id) as u64 => __t6, lateout("dx") __t7, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t6 as u32) };
__val_gu = unsafe { core::mem::transmute::<u32, u32>(__t7 as u32) };
}
instrument_get_user!(__val_gu);
unsafe {
*(pack_id as *mut i32) = __val_gu as i32;
}
__v13 = __ret_gu as i64;
__v12 = __v13;
return __v12 as i32;
}
}
// no valid header was passed, so ignore the pack_id
unsafe {
*(pack_id as *mut i32) = 1i32.wrapping_neg();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_read")]
extern "C" fn sg_read(filp: *mut file, mut buf: *mut u8, mut count: u64, ppos: *mut i64) -> i64 {
let mut __slate_compound_32: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_34: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __v9: bool = unsafe { core::mem::zeroed() };
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut __int: i64 = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __v16: bool = unsafe { core::mem::zeroed() };
let mut __v17: bool = unsafe { core::mem::zeroed() };
let mut __ret_1: i32 = unsafe { core::mem::zeroed() };
let mut __v22: bool = unsafe { core::mem::zeroed() };
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut old_hdr: core::mem::MaybeUninit<*mut sg_header> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut sg_header = core::ptr::addr_of_mut!(old_hdr) as *mut *mut sg_header;
let mut hp: core::mem::MaybeUninit<*mut sg_io_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut sg_io_hdr = core::ptr::addr_of_mut!(hp) as *mut *mut sg_io_hdr;
let mut busy: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut bool = core::ptr::addr_of_mut!(busy) as *mut bool;
let mut req_pack_id: i32 = unsafe { core::mem::zeroed() };
let mut srp: core::mem::MaybeUninit<*mut sg_request> = core::mem::MaybeUninit::uninit();
let __slate_slot_29: *mut *mut sg_request = core::ptr::addr_of_mut!(srp) as *mut *mut sg_request;
let mut sfp: core::mem::MaybeUninit<*mut sg_fd> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut *mut sg_fd = core::ptr::addr_of_mut!(sfp) as *mut *mut sg_fd;
let mut sdp: core::mem::MaybeUninit<*mut sg_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut *mut sg_device = core::ptr::addr_of_mut!(sdp) as *mut *mut sg_device;
unsafe {
let __slate_auto_bytes_2 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_29 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
req_pack_id = 1i32.wrapping_neg();
let __slate_auto_bytes_5 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_6 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_7 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_8 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 4usize);
// This could cause a response to be stranded. Close the associated
// file descriptor to free up any resources being held.
*__slate_slot_24 = sg_check_file_access(filp, c"sg_read".as_ptr() as *const u8);
if *__slate_slot_24 != 0 {
return *__slate_slot_24 as i64;
} else {
let __v23: *mut sg_fd = (*filp).private_data as *mut sg_fd;
*__slate_slot_30 = __v23;
if __v23.is_null() {
__v22 = true;
} else {
let __v21: *mut sg_device = (*(*__slate_slot_30)).parentdp;
*__slate_slot_31 = __v21;
__v22 = __v21.is_null();
}
if __v22 {
return 6i32.wrapping_neg() as i64;
} else {
SCSI_LOG_TIMEOUT!();
if (*(*__slate_slot_30)).force_packid != 0 {
*__slate_slot_24 = get_sg_io_pack_id(&mut (*core::ptr::addr_of_mut!(req_pack_id)), buf as *mut (), count);
}
if *__slate_slot_24 != 0 {
return *__slate_slot_24 as i64;
} else {
*__slate_slot_29 = sg_get_rq_mark(*__slate_slot_30, req_pack_id, &mut (*core::ptr::addr_of_mut!(*__slate_slot_27)));
if (*__slate_slot_29).is_null() {
// now wait on packet to arrive
if (*filp).f_flags & 2048u32 != 0 {
return 11i32.wrapping_neg() as i64;
} else {
__ret_1 = 0;
might_sleep!();
let __v18: *mut sg_request = sg_get_rq_mark(*__slate_slot_30, req_pack_id, &mut (*core::ptr::addr_of_mut!(*__slate_slot_27)));
*__slate_slot_29 = __v18;
if !__v18.is_null() {
__v17 = true;
} else {
if !(*__slate_slot_27) {
__v16 = atomic_read(core::ptr::addr_of_mut!((*(*__slate_slot_31)).detaching) as *const __SlateRecord_types_h_190) != 0;
} else {
__v16 = false;
}
__v17 = __v16;
}
if !__v17 {
let __slate_auto_bytes = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = 0;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_14), if false { 1i32 } else { 0i32 });
'__join_29: {
'__join_30: {
loop {
__int = prepare_to_wait_event(core::ptr::addr_of_mut!((*(*__slate_slot_30)).read_wait), core::ptr::addr_of_mut!(*__slate_slot_14), 1i32);
let __v11: *mut sg_request = sg_get_rq_mark(*__slate_slot_30, req_pack_id, &mut (*core::ptr::addr_of_mut!(*__slate_slot_27)));
*__slate_slot_29 = __v11;
if !__v11.is_null() {
__v10 = true;
} else {
if !(*__slate_slot_27) {
__v9 = atomic_read(core::ptr::addr_of_mut!((*(*__slate_slot_31)).detaching) as *const __SlateRecord_types_h_190) != 0;
} else {
__v9 = false;
}
__v10 = __v9;
}
if __v10 {
break '__join_30;
} else {
if ___wait_is_interruptible!(1i32) && __int != 0 {
break;
} else {
schedule();
let __v8: *mut sg_request = sg_get_rq_mark(*__slate_slot_30, req_pack_id, &mut (*core::ptr::addr_of_mut!(*__slate_slot_27)));
*__slate_slot_29 = __v8;
if !__v8.is_null() {
__v7 = true;
} else {
if !(*__slate_slot_27) {
__v6 = atomic_read(core::ptr::addr_of_mut!((*(*__slate_slot_31)).detaching) as *const __SlateRecord_types_h_190) != 0;
} else {
__v6 = false;
}
__v7 = __v6;
}
if __v7 {
break '__join_30;
}
}
}
}
__ret = __int;
break '__join_29;
}
finish_wait(core::ptr::addr_of_mut!((*(*__slate_slot_30)).read_wait), core::ptr::addr_of_mut!(*__slate_slot_14));
}
__ret_1 = __ret as i32;
}
*__slate_slot_24 = __ret_1;
if (*__slate_slot_29).is_null() {
return (if *__slate_slot_24 != 0 { *__slate_slot_24 } else { 19i32.wrapping_neg() }) as i64;
}
// signal or detaching
}
}
if (*(*__slate_slot_29)).header.interface_id != 0 {
return sg_new_read(*__slate_slot_30, buf, count, *__slate_slot_29);
} else {
*__slate_slot_26 = core::ptr::addr_of_mut!((*(*__slate_slot_29)).header);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_3 = _kzalloc_noprof(core::mem::size_of::<sg_header>() as u64, 3264u32, *{
__slate_compound_32 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_32)
});
alloc_tag_restore!();
} else {
*__slate_slot_3 = _kzalloc_noprof(core::mem::size_of::<sg_header>() as u64, 3264u32, *{
__slate_compound_34 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_34)
});
}
*__slate_slot_25 = *__slate_slot_3 as *mut sg_header;
if (*__slate_slot_25).is_null() {
return 12i32.wrapping_neg() as i64;
} else {
(*(*__slate_slot_25)).reply_len = (*(*__slate_slot_26)).timeout as i32;
(*(*__slate_slot_25)).pack_len = (*(*__slate_slot_25)).reply_len;
// old, strange behaviour
(*(*__slate_slot_25)).pack_id = (*(*__slate_slot_26)).pack_id;
let __t1 = (if ((*(*__slate_slot_29)).data.cmd_opcode as i32) >= 192 && 12 == ((*(*__slate_slot_26)).cmd_len as i32) { 1i32 } else { 0i32 }) as u32;
(*(*__slate_slot_25)).__slate_bits_0.__set_twelve_byte(__t1);
let __t2 = (*(*__slate_slot_26)).masked_status as u32;
(*(*__slate_slot_25)).__slate_bits_0.__set_target_status(__t2);
let __t3 = (*(*__slate_slot_26)).host_status as u32;
(*(*__slate_slot_25)).__slate_bits_0.__set_host_status(__t3);
let __t4 = (*(*__slate_slot_26)).driver_status as u32;
(*(*__slate_slot_25)).__slate_bits_0.__set_driver_status(__t4);
if 1 & ((*(*__slate_slot_26)).masked_status as i32) != 0 || (*((*(*__slate_slot_29)).sense_b.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) & 112 == 112 {
(*(*__slate_slot_25)).__slate_bits_0.__set_driver_status(8u32);
memcpy(((*(*__slate_slot_25)).sense_buffer.as_mut_ptr() as *mut u8) as *mut (), ((*(*__slate_slot_29)).sense_b.as_mut_ptr() as *mut u8) as *const (), core::mem::size_of::<[u8; 16]>() as u64);
}
'__join_10: {
'__join_15: {
'__join_14: {
let __t0: i32 = (*(*__slate_slot_26)).host_status as i32;
if __t0 == 0 {
break '__join_15;
} else {
if __t0 == 10 {
break '__join_15;
} else {
if __t0 == 11 {
break '__join_15;
} else {
if __t0 == 1 {
break '__join_14;
} else {
if __t0 == 2 {
break '__join_14;
} else {
if __t0 == 3 {
break '__join_14;
} else {
if __t0 != 4 {
if __t0 != 5 {
if __t0 != 6 {
if __t0 != 8 {
if __t0 != 9 {
if __t0 == 7 {
(*(*__slate_slot_25)).result = if (*((*(*__slate_slot_29)).sense_b.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) == 0 && ((*(*__slate_slot_26)).masked_status as i32) == 0 { 0 } else { 5 };
break '__join_10;
} else {
(*(*__slate_slot_25)).result = 5;
break '__join_10;
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
(*(*__slate_slot_25)).result = 5;
break '__join_10;
}
(*(*__slate_slot_25)).result = 16;
break '__join_10;
}
(*(*__slate_slot_25)).result = 0;
}
'__join_0: {
// Now copy the result back to the user buffer.
if count >= (core::mem::size_of::<sg_header>() as u64) {
if copy_to_user(buf as *mut (), *__slate_slot_25 as *const (), core::mem::size_of::<sg_header>() as u64) != 0 {
*__slate_slot_24 = 14i32.wrapping_neg();
break '__join_0;
} else {
let __v1: *mut u8 = buf;
buf = __v1.wrapping_offset(core::mem::size_of::<sg_header>() as isize);
if count > ((*(*__slate_slot_25)).reply_len as u64) {
count = (*(*__slate_slot_25)).reply_len as u64;
}
if count > (core::mem::size_of::<sg_header>() as u64) {
if sg_read_oxfer(*__slate_slot_29, buf, count.wrapping_sub(core::mem::size_of::<sg_header>() as u64) as i32) != 0 {
*__slate_slot_24 = 14i32.wrapping_neg();
break '__join_0;
}
}
}
} else {
count = (if (*(*__slate_slot_25)).result == 0 { 0i32 } else { 5i32.wrapping_neg() }) as u64;
}
sg_finish_rem_req(*__slate_slot_29);
sg_remove_request(*__slate_slot_30, *__slate_slot_29);
*__slate_slot_24 = count as i32;
}
kfree(*__slate_slot_25 as *const ());
return *__slate_slot_24 as i64;
}
}
}
}
}
}
// This setup of 'result' is for backward compatibility and is best
// ignored by the user who should use target, host + driver status
}

fn sg_new_read(sfp: *mut sg_fd, buf: *mut u8, count: u64, srp: *mut sg_request) -> i64 {
let mut __v1: u64 = unsafe { core::mem::zeroed() };
let mut __v2: i32 = unsafe { core::mem::zeroed() };
let mut sb_len: i32 = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut err2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(err2) as *mut i32;
let mut err: i32 = unsafe { core::mem::zeroed() };
let mut hp: *mut sg_io_hdr = unsafe { core::mem::zeroed() };
unsafe {
'__join_3: {
hp = core::ptr::addr_of_mut!((*srp).header);
err = 0;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if in_compat_syscall() {
if count < (core::mem::size_of::<compat_sg_io_hdr>() as u64) {
err = 22i32.wrapping_neg();
break '__join_3;
}
} else {
if count < (core::mem::size_of::<sg_io_hdr>() as u64) {
err = 22i32.wrapping_neg();
break '__join_3;
}
}
(*hp).sb_len_wr = 0;
if ((*hp).mx_sb_len as i32) > 0 && !unsafe { (*hp).sbp }.is_null() {
if 1 & ((*hp).masked_status as i32) != 0 || (*((*srp).sense_b.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) & 112 == 112 {
sb_len = 96;
sb_len = if ((*hp).mx_sb_len as i32) > sb_len { sb_len } else { (*hp).mx_sb_len as i32 };
*__slate_slot_7 = 8i32.wrapping_add(*((*srp).sense_b.as_mut_ptr() as *mut u8).wrapping_offset(7 as isize) as i32); // Additional sense length field
*__slate_slot_7 = if *__slate_slot_7 > sb_len { sb_len } else { *__slate_slot_7 };
if copy_to_user((*hp).sbp, ((*srp).sense_b.as_mut_ptr() as *mut u8) as *const (), *__slate_slot_7 as u64) != 0 {
err = 14i32.wrapping_neg();
break '__join_3;
} else {
(*hp).driver_status = 8;
(*hp).sb_len_wr = *__slate_slot_7 as u8;
}
}
}
if (*hp).masked_status != 0 || (*hp).host_status != 0 || (*hp).driver_status != 0 {
(*hp).info = (*hp).info | 1;
}
err = put_sg_io_hdr(hp as *const sg_io_hdr, buf as *mut ());
}
*__slate_slot_8 = sg_finish_rem_req(srp);
sg_remove_request(sfp, srp);
__v2 = err;
if __v2 != 0 {
__v1 = __v2 as u64;
} else {
let __v0: i32 = *__slate_slot_8;
__v1 = if __v0 != 0 { __v0 as u64 } else { count };
}
return __v1 as i64;
}
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_write")]
extern "C" fn sg_write(filp: *mut file, mut buf: *const u8, count: u64, ppos: *mut i64) -> i64 {
let mut mxsize: i32 = 0;
let mut cmd_size: i32 = 0;
let mut k: i32 = 0;
let mut input_size: i32 = 0;
let mut blocking: i32 = 0;
let mut opcode: u8 = 0;
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let mut srp: *mut sg_request = unsafe { core::mem::zeroed() };
let mut old_hdr: sg_header = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(old_hdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 36usize);
}
let mut cmnd: __SlateAlign16<[u8; 252]> = __SlateAlign16([0u8; 252]);
let retval: i32 = sg_check_file_access(filp, c"sg_write".as_ptr() as *const u8);
if retval != 0 {
return retval as i64;
}
let __v0: *mut sg_fd = (unsafe { (*filp).private_data }) as *mut sg_fd;
sfp = __v0;
let __v1: bool = if __v0.is_null() {
true
} else {
let __v2: *mut sg_device = unsafe { (*sfp).parentdp };
sdp = __v2;
__v2.is_null()
};
if __v1 {
return 6i32.wrapping_neg() as i64;
}
SCSI_LOG_TIMEOUT!();
if atomic_read((unsafe { core::ptr::addr_of_mut!((*sdp).detaching) }) as *const __SlateRecord_types_h_190) != 0 {
return 19i32.wrapping_neg() as i64;
}
let __v4: bool = (unsafe { (*filp).f_flags }) & 2048u32 != 0 || (unsafe { scsi_block_when_processing_errors((*sdp).device) }) != 0;
if !__v4 {
return 6i32.wrapping_neg() as i64;
}
if count < (core::mem::size_of::<sg_header>() as u64) {
return 5i32.wrapping_neg() as i64;
}
if copy_from_user(core::ptr::addr_of_mut!(old_hdr) as *mut (), buf as *const (), core::mem::size_of::<sg_header>() as u64) != 0 {
return 14i32.wrapping_neg() as i64;
}
blocking = ((unsafe { (*filp).f_flags }) & 2048u32 == 0) as i32;
if old_hdr.reply_len < 0 {
return sg_new_write(sfp, filp, buf, count, blocking, 0i32, 0i32, core::ptr::null_mut::<*mut sg_request>());
}
if count < (core::mem::size_of::<sg_header>() as u64).wrapping_add(6) {
return 5i32.wrapping_neg() as i64;
}
// The minimum scsi command length is 6 bytes.
let __v5: *const u8 = buf;
buf = __v5.wrapping_offset(core::mem::size_of::<sg_header>() as isize);
let mut __v7: i64;
might_fault();
let mut __v8: i64;
let mut __ret_gu: i32 = 0;
let mut __val_gu: u8 = 0;
__chk_user_ptr!();
unsafe {
let __t0: u64;
let __t1: u32;
core::arch::asm!("call __get_user_{0}", const 1 as u64, inlateout("ax") buf as u64 => __t0, lateout("dx") __t1, options(att_syntax));
__ret_gu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
__val_gu = unsafe { core::mem::transmute::<u8, u8>(__t1 as u8) };
}
instrument_get_user__2!(__val_gu);
opcode = __val_gu;
__v8 = __ret_gu as i64;
__v7 = __v8;
if __v7 != 0 {
return 14i32.wrapping_neg() as i64;
}
let __v11: *mut sg_request = sg_add_request(sfp);
srp = __v11;
if __v11.is_null() {
SCSI_LOG_TIMEOUT!();
return 33i32.wrapping_neg() as i64;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sfp).f_mutex)) };
if ((unsafe { (*sfp).next_cmd_len }) as i32) > 0 {
cmd_size = (unsafe { (*sfp).next_cmd_len }) as i32;
unsafe {
(*sfp).next_cmd_len = 0;
}
// reset so only this write() effected
} else {
cmd_size = COMMAND_SIZE!(opcode) as i32; // based on SCSI command group
if (opcode as i32) >= 192 && (old_hdr.__slate_bits_0.__get_twelve_byte() as i32) != 0 {
cmd_size = 12;
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sfp).f_mutex)) };
SCSI_LOG_TIMEOUT!();
// Determine buffer size.
input_size = count.wrapping_sub(cmd_size as u64) as i32;
mxsize = if input_size > old_hdr.reply_len { input_size } else { old_hdr.reply_len };
mxsize = (mxsize as u64).wrapping_sub(core::mem::size_of::<sg_header>() as u64) as i32;
input_size = (input_size as u64).wrapping_sub(core::mem::size_of::<sg_header>() as u64) as i32;
if input_size < 0 {
sg_remove_request(sfp, srp);
return 5i32.wrapping_neg() as i64; // User did not pass enough bytes for this command.
}
let hp: *mut sg_io_hdr = unsafe { core::ptr::addr_of_mut!((*srp).header) };
unsafe {
(*hp).interface_id = 0;
}
// indicator of old interface tunnelled
unsafe {
(*hp).cmd_len = cmd_size as u8;
}
unsafe {
(*hp).iovec_count = 0;
}
unsafe {
(*hp).mx_sb_len = 0;
}
if input_size > 0 {
unsafe {
(*hp).dxfer_direction = if (old_hdr.reply_len as u64) > (core::mem::size_of::<sg_header>() as u64) { 4i32.wrapping_neg() } else { 2i32.wrapping_neg() };
}
} else {
unsafe {
(*hp).dxfer_direction = if mxsize > 0 { 3i32.wrapping_neg() } else { 1i32.wrapping_neg() };
}
}
unsafe {
(*hp).dxfer_len = mxsize as u32;
}
if (unsafe { (*hp).dxfer_direction }) == 2i32.wrapping_neg() || (unsafe { (*hp).dxfer_direction }) == 4i32.wrapping_neg() {
unsafe {
(*hp).dxferp = (buf as *mut u8).wrapping_offset(cmd_size as isize) as *mut ();
}
} else {
unsafe {
(*hp).dxferp = core::ptr::null_mut::<()>();
}
}
unsafe {
(*hp).sbp = core::ptr::null_mut::<()>();
}
unsafe {
(*hp).timeout = old_hdr.reply_len as u32;
}
// structure abuse ...
unsafe {
(*hp).flags = input_size as u32;
}
// structure abuse ...
unsafe {
(*hp).pack_id = old_hdr.pack_id;
}
unsafe {
(*hp).usr_ptr = core::ptr::null_mut::<()>();
}
if copy_from_user(cmnd.0.as_mut_ptr() as *mut (), buf as *const (), cmd_size as u64) != 0 {
sg_remove_request(sfp, srp);
return 14i32.wrapping_neg() as i64;
}
// SG_DXFER_TO_FROM_DEV is functionally equivalent to SG_DXFER_FROM_DEV,
// but is is possible that the app intended SG_DXFER_TO_DEV, because there
// is a non-zero input_size, so emit a warning.
if (unsafe { (*hp).dxfer_direction }) == 4i32.wrapping_neg() {
if __ratelimit!(unsafe { core::ptr::addr_of_mut!(_rs) }) != 0 {
let mut __v18: i32;
while false {
}
__v18 = unsafe { _printk(c"\x014sg_write: data in/out %d/%d bytes for SCSI command 0x%x-- guessing data in;\n   program %s not setting count and/or reply_len properly\n".as_ptr() as *const u8, old_hdr.reply_len.wrapping_sub(core::mem::size_of::<sg_header>() as i32), input_size, cmnd.0[0] as u32, (*get_current()).comm.as_mut_ptr() as *mut u8) };
}
}
k = sg_common_write(sfp, srp, cmnd.0.as_mut_ptr() as *mut u8, unsafe { (*sfp).timeout }, blocking);
(if k < 0 { k as u64 } else { count }) as i64
}

static mut _rs: ratelimit_state = {
let mut __t0: ratelimit_state = unsafe { core::mem::zeroed() };
__t0.lock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0.interval = 5000i32;
__t0.burst = 10i32;
__t0.flags = 0u32;
__t0
};

fn sg_new_write(sfp: *mut sg_fd, file: *mut file, buf: *const u8, count: u64, blocking: i32, read_only: i32, sg_io_owned: i32, o_srp: *mut *mut sg_request) -> i64 {
let mut k: i32 = 0;
let mut srp: *mut sg_request = unsafe { core::mem::zeroed() };
let mut cmnd: __SlateAlign16<[u8; 252]> = __SlateAlign16([0u8; 252]);
let mut timeout: i32 = 0;
let mut ul_timeout: u64 = 0;
if count < (core::mem::size_of::<sg_io_hdr>() as u64) {
return 22i32.wrapping_neg() as i64;
}
unsafe {
(*sfp).cmd_q = 1;
}
// when sg_io_hdr seen, set command queuing on
let __v0: *mut sg_request = sg_add_request(sfp);
srp = __v0;
if __v0.is_null() {
SCSI_LOG_TIMEOUT!();
return 33i32.wrapping_neg() as i64;
}
unsafe {
(*srp).sg_io_owned = sg_io_owned as u8;
}
let hp: *mut sg_io_hdr = unsafe { core::ptr::addr_of_mut!((*srp).header) };
if (unsafe { get_sg_io_hdr(hp, buf as *const ()) }) != 0 {
sg_remove_request(sfp, srp);
return 14i32.wrapping_neg() as i64;
}
unsafe {
(*hp).duration = jiffies_to_msecs(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)));
}
if (unsafe { (*hp).interface_id }) != 83 {
sg_remove_request(sfp, srp);
return 38i32.wrapping_neg() as i64;
}
if (unsafe { (*hp).flags }) & 4 != 0 {
if (unsafe { (*hp).dxfer_len }) > unsafe { (*sfp).reserve.bufflen } {
sg_remove_request(sfp, srp);
return 12i32.wrapping_neg() as i64; // MMAP_IO size must fit in reserve buffer
}
if (unsafe { (*hp).flags }) & 1 != 0 {
sg_remove_request(sfp, srp);
return 22i32.wrapping_neg() as i64; // either MMAP_IO or DIRECT_IO (not both)
}
if (unsafe { (*sfp).res_in_use }) != 0 {
sg_remove_request(sfp, srp);
return 16i32.wrapping_neg() as i64; // reserve buffer already being used
}
}
ul_timeout = msecs_to_jiffies(unsafe { (*srp).header.timeout });
timeout = (if ul_timeout < 2147483647u64 { ul_timeout } else { 2147483647u64 }) as i32;
if unsafe { (*hp).cmdp }.is_null() || ((unsafe { (*hp).cmd_len }) as i32) < 6 || (((unsafe { (*hp).cmd_len }) as i32) as u64) > (core::mem::size_of::<[u8; 252]>() as u64) {
sg_remove_request(sfp, srp);
return 90i32.wrapping_neg() as i64;
}
if copy_from_user(cmnd.0.as_mut_ptr() as *mut (), (unsafe { (*hp).cmdp }) as *const (), (unsafe { (*hp).cmd_len }) as u64) != 0 {
sg_remove_request(sfp, srp);
return 14i32.wrapping_neg() as i64;
}
let __v2: bool = read_only != 0 && sg_allow_access(file, cmnd.0.as_mut_ptr() as *mut u8) != 0;
if __v2 {
sg_remove_request(sfp, srp);
return 1i32.wrapping_neg() as i64;
}
k = sg_common_write(sfp, srp, cmnd.0.as_mut_ptr() as *mut u8, timeout, blocking);
if k < 0 {
return k as i64;
}
if !o_srp.is_null() {
unsafe {
*o_srp = srp;
}
}
count as i64
}

fn sg_common_write(sfp: *mut sg_fd, srp: *mut sg_request, cmnd: *mut u8, timeout: i32, blocking: i32) -> i32 {
let mut k: i32 = 0;
let mut at_head: i32 = 0;
let sdp: *mut sg_device = unsafe { (*sfp).parentdp };
let hp: *mut sg_io_hdr = unsafe { core::ptr::addr_of_mut!((*srp).header) };
unsafe {
(*srp).data.cmd_opcode = *cmnd.wrapping_offset(0 as isize);
}
// hold opcode of command
unsafe {
(*hp).status = 0;
}
unsafe {
(*hp).masked_status = 0;
}
unsafe {
(*hp).msg_status = 0;
}
unsafe {
(*hp).info = 0;
}
unsafe {
(*hp).host_status = 0;
}
unsafe {
(*hp).driver_status = 0;
}
unsafe {
(*hp).resid = 0;
}
SCSI_LOG_TIMEOUT!();
if (unsafe { (*hp).dxfer_len }) >= 268435456 {
sg_remove_request(sfp, srp);
return 22i32.wrapping_neg();
}
k = sg_start_req(srp, cmnd);
if k != 0 {
SCSI_LOG_TIMEOUT!();
sg_finish_rem_req(srp);
sg_remove_request(sfp, srp);
return k; // probably out of space --> ENOMEM
}
if atomic_read((unsafe { core::ptr::addr_of_mut!((*sdp).detaching) }) as *const __SlateRecord_types_h_190) != 0 {
if !unsafe { (*srp).bio }.is_null() {
unsafe { blk_mq_free_request((*srp).rq) };
unsafe {
(*srp).rq = core::ptr::null_mut::<request>();
}
}
sg_finish_rem_req(srp);
sg_remove_request(sfp, srp);
return 19i32.wrapping_neg();
}
if (unsafe { (*hp).interface_id }) != 0 && 16 & unsafe { (*hp).flags } != 0 {
at_head = 0;
} else {
at_head = 1;
}
// v3 (or later) interface
unsafe {
(*(*srp).rq).timeout = timeout as u32;
}
kref_get(unsafe { core::ptr::addr_of_mut!((*sfp).f_ref) }); // sg_rq_end_io() does kref_put().
unsafe {
(*(*srp).rq).end_io = Some(sg_rq_end_io);
}
unsafe { blk_execute_rq_nowait((*srp).rq, at_head != 0) };
0
}

fn srp_done(sfp: *mut sg_fd, srp: *mut sg_request) -> i32 {
let mut flags: u64 = 0;
let mut ret: i32 = 0;
read_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, flags);
ret = (unsafe { (*srp).done }) as i32;
read_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, flags);
ret
}

fn max_sectors_bytes(q: *mut request_queue) -> i32 {
let mut max_sectors: u32 = queue_max_sectors(q as *const request_queue);
max_sectors = min_t__2!(u32, max_sectors, 4194303i32);
(max_sectors << 9i32) as i32
}

fn sg_fill_request_table(sfp: *mut sg_fd, rinfo: *mut sg_req_info) {
let mut srp: *mut sg_request = unsafe { core::mem::zeroed() };
let mut val: i32 = 0;
let mut ms: u32 = 0;
srp = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) });
while list_entry_is_head!(srp, unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) }, entry) == 0 {
if val >= 16 {
break;
}
unsafe {
(*rinfo.wrapping_offset(val as isize)).req_state = ((*srp).done as i32).wrapping_add(1) as u8;
}
unsafe {
(*rinfo.wrapping_offset(val as isize)).problem = ((*srp).header.masked_status != 0 || (*srp).header.host_status != 0 || (*srp).header.driver_status != 0) as u8;
}
if (unsafe { (*srp).done }) != 0 {
unsafe {
(*rinfo.wrapping_offset(val as isize)).duration = (*srp).header.duration;
}
} else {
ms = jiffies_to_msecs(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) });
unsafe {
(*rinfo.wrapping_offset(val as isize)).duration = if ms > (*srp).header.duration { ms.wrapping_sub((*srp).header.duration) } else { 0 };
}
}
unsafe {
(*rinfo.wrapping_offset(val as isize)).orphan = (*srp).orphan;
}
unsafe {
(*rinfo.wrapping_offset(val as isize)).sg_io_owned = (*srp).sg_io_owned;
}
unsafe {
(*rinfo.wrapping_offset(val as isize)).pack_id = (*srp).header.pack_id;
}
unsafe {
(*rinfo.wrapping_offset(val as isize)).usr_ptr = (*srp).header.usr_ptr;
}
val = val.wrapping_add(1);
srp = list_next_entry__3!(srp, entry);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct compat_sg_req_info {
    req_state: u8,
    orphan: u8,
    sg_io_owned: u8,
    problem: u8,
    pack_id: i32,
    usr_ptr: u32,
    duration: u32,
    unused: i32,
    // used by SG_GET_REQUEST_TABLE ioctl()
}


fn put_compat_request_table(o: *mut compat_sg_req_info, rinfo: *mut sg_req_info) -> i32 {
for i in 0i32..16i32 {
let __v0: bool = if copy_to_user(o.wrapping_offset(i as isize) as *mut (), rinfo.wrapping_offset(i as isize) as *const (), offsetof__35!(sg_req_info)) != 0 {
true
} else {
let mut __v1: i64;
might_fault();
let mut __v2: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: u32 = 0;
let mut __x: u32 = ((unsafe { (*rinfo.wrapping_offset(i as isize)).usr_ptr }) as u64) as u32;
let __ptr: *mut u32 = unsafe { core::ptr::addr_of_mut!((*o.wrapping_offset(i as isize)).usr_ptr) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t0, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
instrument_put_user__2!(__x, __ptr);
__v2 = __ret_pu as i64;
__v1 = __v2;
__v1 != 0
};
let __v3: bool = if __v0 {
true
} else {
let mut __v4: i64;
might_fault();
let mut __v5: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: u32 = 0;
let mut __x: u32 = unsafe { (*rinfo.wrapping_offset(i as isize)).duration };
let __ptr: *mut u32 = unsafe { core::ptr::addr_of_mut!((*o.wrapping_offset(i as isize)).duration) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t1: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t1, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
instrument_put_user__2!(__x, __ptr);
__v5 = __ret_pu as i64;
__v4 = __v5;
__v4 != 0
};
let __v6: bool = if __v3 {
true
} else {
let mut __v7: i64;
might_fault();
let mut __v8: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: i32 = 0;
let mut __x: i32 = unsafe { (*rinfo.wrapping_offset(i as isize)).unused };
let __ptr: *mut i32 = unsafe { core::ptr::addr_of_mut!((*o.wrapping_offset(i as isize)).unused) };
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t2: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t2, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
instrument_put_user!(__x, __ptr);
__v8 = __ret_pu as i64;
__v7 = __v8;
__v7 != 0
};
if __v6 {
return 14i32.wrapping_neg();
}
}
0
}

fn sg_ioctl_common(filp: *mut file, sdp: *mut sg_device, sfp: *mut sg_fd, cmd_in: u32, p: *mut ()) -> i64 {
let mut __slate_compound_138: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_140: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __x: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(__val_pu) as *mut i32;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __val_gu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u32 = core::ptr::addr_of_mut!(__val_gu) as *mut u32;
let mut __ret_gu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(__ret_gu) as *mut i32;
let mut __x_1: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(__val_pu_1) as *mut i32;
let mut __ptr_pu_1: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_1) as *mut *mut ();
let mut __ret_pu_1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_1) as *mut i32;
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut rinfo: core::mem::MaybeUninit<*mut sg_req_info> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut *mut sg_req_info = core::ptr::addr_of_mut!(rinfo) as *mut *mut sg_req_info;
let mut __x_2: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut i32 = core::ptr::addr_of_mut!(__val_pu_2) as *mut i32;
let mut __ptr_pu_2: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_26: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_2) as *mut *mut ();
let mut __ret_pu_2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_27: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_2) as *mut i32;
let mut __x_3: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_3: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut i32 = core::ptr::addr_of_mut!(__val_pu_3) as *mut i32;
let mut __ptr_pu_3: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_3) as *mut *mut ();
let mut __ret_pu_3: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_3) as *mut i32;
let mut __val_gu_1: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut u32 = core::ptr::addr_of_mut!(__val_gu_1) as *mut u32;
let mut __ret_gu_1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut i32 = core::ptr::addr_of_mut!(__ret_gu_1) as *mut i32;
let mut __x_4: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_4: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_43: *mut i32 = core::ptr::addr_of_mut!(__val_pu_4) as *mut i32;
let mut __ptr_pu_4: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_44: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_4) as *mut *mut ();
let mut __ret_pu_4: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_45: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_4) as *mut i32;
let mut __val_gu_2: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_48: *mut u32 = core::ptr::addr_of_mut!(__val_gu_2) as *mut u32;
let mut __ret_gu_2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_49: *mut i32 = core::ptr::addr_of_mut!(__ret_gu_2) as *mut i32;
let mut __x_5: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_5: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_54: *mut i32 = core::ptr::addr_of_mut!(__val_pu_5) as *mut i32;
let mut __ptr_pu_5: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_55: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_5) as *mut *mut ();
let mut __ret_pu_5: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_56: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_5) as *mut i32;
let mut __val_gu_3: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_59: *mut u32 = core::ptr::addr_of_mut!(__val_gu_3) as *mut u32;
let mut __ret_gu_3: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_60: *mut i32 = core::ptr::addr_of_mut!(__ret_gu_3) as *mut i32;
let mut __x_6: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_6: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_65: *mut i32 = core::ptr::addr_of_mut!(__val_pu_6) as *mut i32;
let mut __ptr_pu_6: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_66: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_6) as *mut *mut ();
let mut __ret_pu_6: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_67: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_6) as *mut i32;
let mut __val_gu_4: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_70: *mut u32 = core::ptr::addr_of_mut!(__val_gu_4) as *mut u32;
let mut __ret_gu_4: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_71: *mut i32 = core::ptr::addr_of_mut!(__ret_gu_4) as *mut i32;
let mut __x_7: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_7: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_76: *mut i32 = core::ptr::addr_of_mut!(__val_pu_7) as *mut i32;
let mut __ptr_pu_7: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_77: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_7) as *mut *mut ();
let mut __ret_pu_7: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_78: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_7) as *mut i32;
let mut __x_8: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_8: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_83: *mut i32 = core::ptr::addr_of_mut!(__val_pu_8) as *mut i32;
let mut __ptr_pu_8: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_84: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_8) as *mut *mut ();
let mut __ret_pu_8: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_85: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_8) as *mut i32;
let mut __x_9: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_9: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_92: *mut i32 = core::ptr::addr_of_mut!(__val_pu_9) as *mut i32;
let mut __ptr_pu_9: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_93: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_9) as *mut *mut ();
let mut __ret_pu_9: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_94: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_9) as *mut i32;
let mut __x_10: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu_10: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_99: *mut i32 = core::ptr::addr_of_mut!(__val_pu_10) as *mut i32;
let mut __ptr_pu_10: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_100: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu_10) as *mut *mut ();
let mut __ret_pu_10: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_101: *mut i32 = core::ptr::addr_of_mut!(__ret_pu_10) as *mut i32;
let mut __val_gu_5: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_104: *mut u32 = core::ptr::addr_of_mut!(__val_gu_5) as *mut u32;
let mut __ret_gu_5: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_105: *mut i32 = core::ptr::addr_of_mut!(__ret_gu_5) as *mut i32;
let mut v: core::mem::MaybeUninit<sg_scsi_id> = core::mem::MaybeUninit::uninit();
let __slate_slot_108: *mut sg_scsi_id = core::ptr::addr_of_mut!(v) as *mut sg_scsi_id;
let mut __x_11: i32 = unsafe { core::mem::zeroed() };
let mut __val_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_111: *mut i32 = core::ptr::addr_of_mut!(__val_pu) as *mut i32;
let mut __ptr_pu: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_112: *mut *mut () = core::ptr::addr_of_mut!(__ptr_pu) as *mut *mut ();
let mut __ret_pu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_113: *mut i32 = core::ptr::addr_of_mut!(__ret_pu) as *mut i32;
let mut __val_gu: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_116: *mut u32 = core::ptr::addr_of_mut!(__val_gu) as *mut u32;
let mut __ret_gu: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_117: *mut i32 = core::ptr::addr_of_mut!(__ret_gu) as *mut i32;
let mut __int: i64 = unsafe { core::mem::zeroed() };
let mut __ret: i64 = unsafe { core::mem::zeroed() };
let mut __wq_entry: core::mem::MaybeUninit<wait_queue_entry> = core::mem::MaybeUninit::uninit();
let __slate_slot_122: *mut wait_queue_entry = core::ptr::addr_of_mut!(__wq_entry) as *mut wait_queue_entry;
let mut __ret_1: i32 = unsafe { core::mem::zeroed() };
let mut iflags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_126: *mut u64 = core::ptr::addr_of_mut!(iflags) as *mut u64;
let mut srp: core::mem::MaybeUninit<*mut sg_request> = core::mem::MaybeUninit::uninit();
let __slate_slot_127: *mut *mut sg_request = core::ptr::addr_of_mut!(srp) as *mut *mut sg_request;
let mut read_only: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_128: *mut i32 = core::ptr::addr_of_mut!(read_only) as *mut i32;
let mut val: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_129: *mut i32 = core::ptr::addr_of_mut!(val) as *mut i32;
let mut result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_130: *mut i32 = core::ptr::addr_of_mut!(result) as *mut i32;
let mut ip: core::mem::MaybeUninit<*mut i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_131: *mut *mut i32 = core::ptr::addr_of_mut!(ip) as *mut *mut i32;
unsafe {
'__join_2: {
core::ptr::write(__slate_slot_131, p as *mut i32);
let __slate_auto_bytes_33 = __slate_slot_130 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_33.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_34 = __slate_slot_129 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_34.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_35 = __slate_slot_128 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_35.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_36 = __slate_slot_127 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_36.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_37 = __slate_slot_126 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_37.add(0usize), 0u8, 8usize);
SCSI_LOG_TIMEOUT!();
*__slate_slot_128 = (2u32 != (*filp).f_flags & 3) as i32;
let __t26: u32 = cmd_in;
if __t26 == 8837 {
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 {
return 19i32.wrapping_neg() as i64;
} else {
if scsi_block_when_processing_errors((*sdp).device) == 0 {
return 6i32.wrapping_neg() as i64;
} else {
*__slate_slot_130 = sg_new_write(sfp, filp, p as *const u8, core::mem::size_of::<sg_io_hdr>() as u64, 1i32, *__slate_slot_128, 1i32, core::ptr::addr_of_mut!(*__slate_slot_127)) as i32;
if *__slate_slot_130 < 0 {
return *__slate_slot_130 as i64;
} else {
__ret_1 = 0;
might_sleep!();
if srp_done(sfp, *__slate_slot_127) == 0 {
let __slate_auto_bytes = __slate_slot_122 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 40usize);
__ret = 0;
init_wait_entry(core::ptr::addr_of_mut!(*__slate_slot_122), if false { 1i32 } else { 0i32 });
'__join_90: {
'__join_91: {
loop {
__int = prepare_to_wait_event(core::ptr::addr_of_mut!((*sfp).read_wait), core::ptr::addr_of_mut!(*__slate_slot_122), 1i32);
if srp_done(sfp, *__slate_slot_127) != 0 {
break '__join_91;
} else {
if ___wait_is_interruptible!(1i32) && __int != 0 {
break;
} else {
schedule();
if srp_done(sfp, *__slate_slot_127) != 0 {
break '__join_91;
}
}
}
}
__ret = __int;
break '__join_90;
}
finish_wait(core::ptr::addr_of_mut!((*sfp).read_wait), core::ptr::addr_of_mut!(*__slate_slot_122));
}
__ret_1 = __ret as i32;
}
*__slate_slot_130 = __ret_1;
write_lock_irq!(core::ptr::addr_of_mut!((*sfp).rq_list_lock));
if (*(*__slate_slot_127)).done != 0 {
(*(*__slate_slot_127)).done = 2;
write_unlock_irq!(core::ptr::addr_of_mut!((*sfp).rq_list_lock));
*__slate_slot_130 = sg_new_read(sfp, p as *mut u8, core::mem::size_of::<sg_io_hdr>() as u64, *__slate_slot_127) as i32;
return (if *__slate_slot_130 < 0 { *__slate_slot_130 } else { 0i32 }) as i64;
} else {
(*(*__slate_slot_127)).orphan = 1;
write_unlock_irq!(core::ptr::addr_of_mut!((*sfp).rq_list_lock));
return *__slate_slot_130 as i64;
}
}
}
}
} else {
if __t26 == 8705 {
might_fault();
let __slate_auto_bytes_32 = __slate_slot_117 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_32.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_116 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__chk_user_ptr!();
let __t24: u64;
let __t25: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") *__slate_slot_131 as u64 => __t24, lateout("dx") __t25, options(att_syntax));
*__slate_slot_117 = unsafe { core::mem::transmute::<u32, i32>(__t24 as u32) };
*__slate_slot_116 = unsafe { core::mem::transmute::<u32, u32>(__t25 as u32) };
instrument_get_user!(*__slate_slot_116);
*__slate_slot_129 = *__slate_slot_116 as i32;
*__slate_slot_130 = (*__slate_slot_117 as i64) as i32;
if *__slate_slot_130 != 0 {
return *__slate_slot_130 as i64;
} else {
if *__slate_slot_129 < 0 {
return 5i32.wrapping_neg() as i64;
} else {
if (*__slate_slot_129 as i64) >= mult_frac__2!(2147483647i64, 100i32, 1000i32) {
*__slate_slot_129 = min_t__2!(i64, mult_frac__2!(2147483647i64, 100i32, 1000i32), 2147483647i32) as i32;
}
(*sfp).timeout_user = *__slate_slot_129;
(*sfp).timeout = mult_frac!(*__slate_slot_129, 1000i32, 100i32);
return 0;
}
}
} else {
if __t26 == 8706 {
return (*sfp).timeout_user as i64;
} else {
if __t26 == 8825 {
return 0;
} else {
if __t26 == 8826 {
might_fault();
let __slate_auto_bytes_30 = __slate_slot_113 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_30.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_31 = __slate_slot_112 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_31.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_111 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_11 = 0;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_112 = __ptr as *mut ();
*__slate_slot_111 = __x_11;
let __t23: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_112 as u64 => __t23, in("ax") *__slate_slot_111 as u32, options(att_syntax));
*__slate_slot_113 = unsafe { core::mem::transmute::<u32, i32>(__t23 as u32) };
instrument_put_user!(__x_11, __ptr);
return *__slate_slot_113 as i64;
} else {
if __t26 == 8822 {
let __slate_auto_bytes = __slate_slot_108 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 32usize);
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 {
return 19i32.wrapping_neg() as i64;
} else {
memset(core::ptr::addr_of_mut!(*__slate_slot_108) as *mut (), 0i32, core::mem::size_of::<sg_scsi_id>() as u64);
(*__slate_slot_108).host_no = (*(*(*sdp).device).host).host_no as i32;
(*__slate_slot_108).channel = (*(*sdp).device).channel as i32;
(*__slate_slot_108).scsi_id = (*(*sdp).device).id as i32;
(*__slate_slot_108).lun = (*(*sdp).device).lun as i32;
(*__slate_slot_108).scsi_type = (*(*sdp).device).r#type as i32;
(*__slate_slot_108).h_cmd_per_lun = (*(*(*sdp).device).host).cmd_per_lun;
(*__slate_slot_108).d_queue_depth = (*(*sdp).device).queue_depth as i16;
if copy_to_user(p, core::ptr::addr_of_mut!(*__slate_slot_108) as *const (), core::mem::size_of::<sg_scsi_id>() as u64) != 0 {
return 14i32.wrapping_neg() as i64;
} else {
return 0;
}
}
} else {
if __t26 == 8827 {
might_fault();
let __slate_auto_bytes_29 = __slate_slot_105 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_29.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_104 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__chk_user_ptr!();
let __t21: u64;
let __t22: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") *__slate_slot_131 as u64 => __t21, lateout("dx") __t22, options(att_syntax));
*__slate_slot_105 = unsafe { core::mem::transmute::<u32, i32>(__t21 as u32) };
*__slate_slot_104 = unsafe { core::mem::transmute::<u32, u32>(__t22 as u32) };
instrument_get_user!(*__slate_slot_104);
*__slate_slot_129 = *__slate_slot_104 as i32;
*__slate_slot_130 = (*__slate_slot_105 as i64) as i32;
if *__slate_slot_130 != 0 {
return *__slate_slot_130 as i64;
} else {
(*sfp).force_packid = (if *__slate_slot_129 != 0 { 1i32 } else { 0i32 }) as u8;
return 0;
}
} else {
if __t26 == 8828 {
read_lock_irqsave!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_126);
*__slate_slot_127 = list_first_entry__3!(core::ptr::addr_of_mut!((*sfp).rq_list));
while list_entry_is_head!(*__slate_slot_127, core::ptr::addr_of_mut!((*sfp).rq_list), entry) == 0 {
if 1 == ((*(*__slate_slot_127)).done as i32) && (*(*__slate_slot_127)).sg_io_owned == 0 {
read_unlock_irqrestore!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_126);
might_fault();
let __slate_auto_bytes_27 = __slate_slot_101 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_27.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_28 = __slate_slot_100 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_28.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_99 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_10 = (*(*__slate_slot_127)).header.pack_id;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_100 = __ptr as *mut ();
*__slate_slot_99 = __x_10;
let __t20: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_100 as u64 => __t20, in("ax") *__slate_slot_99 as u32, options(att_syntax));
*__slate_slot_101 = unsafe { core::mem::transmute::<u32, i32>(__t20 as u32) };
instrument_put_user!(__x_10, __ptr);
return *__slate_slot_101 as i64;
} else {
*__slate_slot_127 = list_next_entry__3!(*__slate_slot_127, entry);
}
}
read_unlock_irqrestore!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_126);
might_fault();
let __slate_auto_bytes_25 = __slate_slot_94 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_25.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_26 = __slate_slot_93 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_26.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_92 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_9 = 1i32.wrapping_neg();
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_93 = __ptr as *mut ();
*__slate_slot_92 = __x_9;
let __t19: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_93 as u64 => __t19, in("ax") *__slate_slot_92 as u32, options(att_syntax));
*__slate_slot_94 = unsafe { core::mem::transmute::<u32, i32>(__t19 as u32) };
instrument_put_user!(__x_9, __ptr);
return *__slate_slot_94 as i64;
} else {
if __t26 == 8829 {
read_lock_irqsave!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_126);
*__slate_slot_129 = 0;
*__slate_slot_127 = list_first_entry__3!(core::ptr::addr_of_mut!((*sfp).rq_list));
while list_entry_is_head!(*__slate_slot_127, core::ptr::addr_of_mut!((*sfp).rq_list), entry) == 0 {
if 1 == ((*(*__slate_slot_127)).done as i32) && (*(*__slate_slot_127)).sg_io_owned == 0 {
let __v89: i32 = *__slate_slot_129;
*__slate_slot_129 = __v89.wrapping_add(1);
}
*__slate_slot_127 = list_next_entry__3!(*__slate_slot_127, entry);
}
read_unlock_irqrestore!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_126);
might_fault();
let __slate_auto_bytes_23 = __slate_slot_85 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_23.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_24 = __slate_slot_84 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_24.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_83 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_8 = *__slate_slot_129;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_84 = __ptr as *mut ();
*__slate_slot_83 = __x_8;
let __t18: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_84 as u64 => __t18, in("ax") *__slate_slot_83 as u32, options(att_syntax));
*__slate_slot_85 = unsafe { core::mem::transmute::<u32, i32>(__t18 as u32) };
instrument_put_user!(__x_8, __ptr);
return *__slate_slot_85 as i64;
} else {
if __t26 == 8831 {
might_fault();
let __slate_auto_bytes_21 = __slate_slot_78 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_21.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_22 = __slate_slot_77 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_22.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_76 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_7 = (*sdp).sg_tablesize;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_77 = __ptr as *mut ();
*__slate_slot_76 = __x_7;
let __t17: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_77 as u64 => __t17, in("ax") *__slate_slot_76 as u32, options(att_syntax));
*__slate_slot_78 = unsafe { core::mem::transmute::<u32, i32>(__t17 as u32) };
instrument_put_user!(__x_7, __ptr);
return *__slate_slot_78 as i64;
} else {
if __t26 == 8821 {
might_fault();
let __slate_auto_bytes_20 = __slate_slot_71 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_20.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_70 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__chk_user_ptr!();
let __t15: u64;
let __t16: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") *__slate_slot_131 as u64 => __t15, lateout("dx") __t16, options(att_syntax));
*__slate_slot_71 = unsafe { core::mem::transmute::<u32, i32>(__t15 as u32) };
*__slate_slot_70 = unsafe { core::mem::transmute::<u32, u32>(__t16 as u32) };
instrument_get_user!(*__slate_slot_70);
*__slate_slot_129 = *__slate_slot_70 as i32;
*__slate_slot_130 = (*__slate_slot_71 as i64) as i32;
if *__slate_slot_130 != 0 {
return *__slate_slot_130 as i64;
} else {
if *__slate_slot_129 < 0 {
return 22i32.wrapping_neg() as i64;
} else {
*__slate_slot_129 = min_t!(i32, *__slate_slot_129, max_sectors_bytes((*(*sdp).device).request_queue));
mutex_lock(core::ptr::addr_of_mut!((*sfp).f_mutex));
if (*__slate_slot_129 as u32) != (*sfp).reserve.bufflen {
if (*sfp).mmap_called != 0 || (*sfp).res_in_use != 0 {
mutex_unlock(core::ptr::addr_of_mut!((*sfp).f_mutex));
return 16i32.wrapping_neg() as i64;
} else {
sg_remove_scat(sfp, core::ptr::addr_of_mut!((*sfp).reserve));
sg_build_reserve(sfp, *__slate_slot_129);
}
}
mutex_unlock(core::ptr::addr_of_mut!((*sfp).f_mutex));
return 0;
}
}
} else {
if __t26 == 8818 {
*__slate_slot_129 = min_t__3!(i32, unsafe { (*sfp).reserve.bufflen }, max_sectors_bytes((*(*sdp).device).request_queue));
might_fault();
let __slate_auto_bytes_18 = __slate_slot_67 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_18.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_19 = __slate_slot_66 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_19.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_65 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_6 = *__slate_slot_129;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_66 = __ptr as *mut ();
*__slate_slot_65 = __x_6;
let __t14: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_66 as u64 => __t14, in("ax") *__slate_slot_65 as u32, options(att_syntax));
*__slate_slot_67 = unsafe { core::mem::transmute::<u32, i32>(__t14 as u32) };
instrument_put_user!(__x_6, __ptr);
return *__slate_slot_67 as i64;
} else {
if __t26 == 8817 {
might_fault();
let __slate_auto_bytes_17 = __slate_slot_60 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_17.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_59 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__chk_user_ptr!();
let __t12: u64;
let __t13: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") *__slate_slot_131 as u64 => __t12, lateout("dx") __t13, options(att_syntax));
*__slate_slot_60 = unsafe { core::mem::transmute::<u32, i32>(__t12 as u32) };
*__slate_slot_59 = unsafe { core::mem::transmute::<u32, u32>(__t13 as u32) };
instrument_get_user!(*__slate_slot_59);
*__slate_slot_129 = *__slate_slot_59 as i32;
*__slate_slot_130 = (*__slate_slot_60 as i64) as i32;
if *__slate_slot_130 != 0 {
return *__slate_slot_130 as i64;
} else {
(*sfp).cmd_q = (if *__slate_slot_129 != 0 { 1i32 } else { 0i32 }) as u8;
return 0;
}
} else {
if __t26 == 8816 {
might_fault();
let __slate_auto_bytes_15 = __slate_slot_56 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_15.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_16 = __slate_slot_55 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_16.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_54 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_5 = (*sfp).cmd_q as i32;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_55 = __ptr as *mut ();
*__slate_slot_54 = __x_5;
let __t11: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_55 as u64 => __t11, in("ax") *__slate_slot_54 as u32, options(att_syntax));
*__slate_slot_56 = unsafe { core::mem::transmute::<u32, i32>(__t11 as u32) };
instrument_put_user!(__x_5, __ptr);
return *__slate_slot_56 as i64;
} else {
if __t26 == 8839 {
might_fault();
let __slate_auto_bytes_14 = __slate_slot_49 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_14.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_48 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__chk_user_ptr!();
let __t9: u64;
let __t10: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") *__slate_slot_131 as u64 => __t9, lateout("dx") __t10, options(att_syntax));
*__slate_slot_49 = unsafe { core::mem::transmute::<u32, i32>(__t9 as u32) };
*__slate_slot_48 = unsafe { core::mem::transmute::<u32, u32>(__t10 as u32) };
instrument_get_user!(*__slate_slot_48);
*__slate_slot_129 = *__slate_slot_48 as i32;
*__slate_slot_130 = (*__slate_slot_49 as i64) as i32;
if *__slate_slot_130 != 0 {
return *__slate_slot_130 as i64;
} else {
(*sfp).keep_orphan = *__slate_slot_129 as u8;
return 0;
}
} else {
if __t26 == 8840 {
might_fault();
let __slate_auto_bytes_12 = __slate_slot_45 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_12.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_13 = __slate_slot_44 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_13.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_43 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_4 = (*sfp).keep_orphan as i32;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_44 = __ptr as *mut ();
*__slate_slot_43 = __x_4;
let __t8: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_44 as u64 => __t8, in("ax") *__slate_slot_43 as u32, options(att_syntax));
*__slate_slot_45 = unsafe { core::mem::transmute::<u32, i32>(__t8 as u32) };
instrument_put_user!(__x_4, __ptr);
return *__slate_slot_45 as i64;
} else {
if __t26 == 8835 {
might_fault();
let __slate_auto_bytes_11 = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_11.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__chk_user_ptr!();
let __t6: u64;
let __t7: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") *__slate_slot_131 as u64 => __t6, lateout("dx") __t7, options(att_syntax));
*__slate_slot_38 = unsafe { core::mem::transmute::<u32, i32>(__t6 as u32) };
*__slate_slot_37 = unsafe { core::mem::transmute::<u32, u32>(__t7 as u32) };
instrument_get_user!(*__slate_slot_37);
*__slate_slot_129 = *__slate_slot_37 as i32;
*__slate_slot_130 = (*__slate_slot_38 as i64) as i32;
if *__slate_slot_130 != 0 {
return *__slate_slot_130 as i64;
} else {
if *__slate_slot_129 > 252 {
return 12i32.wrapping_neg() as i64;
} else {
(*sfp).next_cmd_len = (if *__slate_slot_129 > 0 { *__slate_slot_129 } else { 0i32 }) as u8;
return 0;
}
}
} else {
if __t26 == 8834 {
might_fault();
let __slate_auto_bytes_9 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_9.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_10 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_10.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_3 = sg_version_num;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_33 = __ptr as *mut ();
*__slate_slot_32 = __x_3;
let __t5: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_33 as u64 => __t5, in("ax") *__slate_slot_32 as u32, options(att_syntax));
*__slate_slot_34 = unsafe { core::mem::transmute::<u32, i32>(__t5 as u32) };
instrument_put_user!(__x_3, __ptr);
return *__slate_slot_34 as i64;
} else {
if __t26 == 8841 {
*__slate_slot_129 = if !unsafe { (*sdp).device }.is_null() { 1 } else { 0 }; // faked - we don't have a real access count anymore
might_fault();
let __slate_auto_bytes_7 = __slate_slot_27 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_8 = __slate_slot_26 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_8.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_2 = *__slate_slot_129;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_26 = __ptr as *mut ();
*__slate_slot_25 = __x_2;
let __t4: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_26 as u64 => __t4, in("ax") *__slate_slot_25 as u32, options(att_syntax));
*__slate_slot_27 = unsafe { core::mem::transmute::<u32, i32>(__t4 as u32) };
instrument_put_user!(__x_2, __ptr);
return *__slate_slot_27 as i64;
} else {
if __t26 == 8838 {
let __slate_auto_bytes_5 = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_6 = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_18 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_19 = _kmalloc_array_noprof(16u64, core::mem::size_of::<sg_req_info>() as u64, 3520u32, *{
__slate_compound_138 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_138)
});
alloc_tag_restore!();
} else {
*__slate_slot_19 = _kmalloc_array_noprof(16u64, core::mem::size_of::<sg_req_info>() as u64, 3520u32, *{
__slate_compound_140 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_140)
});
}
*__slate_slot_22 = *__slate_slot_19 as *mut sg_req_info;
if (*__slate_slot_22).is_null() {
return 12i32.wrapping_neg() as i64;
} else {
read_lock_irqsave!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_126);
sg_fill_request_table(sfp, *__slate_slot_22);
read_unlock_irqrestore!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_126);
if in_compat_syscall() {
*__slate_slot_130 = put_compat_request_table(p as *mut compat_sg_req_info, *__slate_slot_22);
} else {
*__slate_slot_130 = copy_to_user(p, *__slate_slot_22 as *const (), (core::mem::size_of::<sg_req_info>() as u64).wrapping_mul(16)) as i32;
}
*__slate_slot_130 = if *__slate_slot_130 != 0 { 14i32.wrapping_neg() } else { 0 };
kfree(*__slate_slot_22 as *const ());
return *__slate_slot_130 as i64;
}
} else {
if __t26 == 8707 {
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 {
return 19i32.wrapping_neg() as i64;
} else {
might_fault();
let __slate_auto_bytes_3 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x_1 = (*(*(*(*sdp).device).host).hostt).__slate_bits_0.__get_emulated() as i32;
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_14 = __ptr as *mut ();
*__slate_slot_13 = __x_1;
let __t3: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_14 as u64 => __t3, in("ax") *__slate_slot_13 as u32, options(att_syntax));
*__slate_slot_15 = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
instrument_put_user!(__x_1, __ptr);
return *__slate_slot_15 as i64;
}
} else {
if __t26 == 1 {
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 {
return 19i32.wrapping_neg() as i64;
} else {
return scsi_ioctl((*sdp).device, (*filp).f_mode & 2u32 != 0, cmd_in as i32, p) as i64;
}
} else {
if __t26 == 8830 {
might_fault();
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__chk_user_ptr!();
let __t1: u64;
let __t2: u32;
core::arch::asm!("call __get_user_{0}", const 4 as u64, inlateout("ax") *__slate_slot_131 as u64 => __t1, lateout("dx") __t2, options(att_syntax));
*__slate_slot_8 = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
*__slate_slot_7 = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
instrument_get_user!(*__slate_slot_7);
*__slate_slot_129 = *__slate_slot_7 as i32;
*__slate_slot_130 = (*__slate_slot_8 as i64) as i32;
if *__slate_slot_130 != 0 {
return *__slate_slot_130 as i64;
} else {
(*sdp).sgdebug = *__slate_slot_129 as u8;
return 0;
}
} else {
if __t26 == 4711 {
might_fault();
let __slate_auto_bytes_38 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_38.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__x = max_sectors_bytes((*(*sdp).device).request_queue);
let __ptr: *mut i32 = *__slate_slot_131;
__chk_user_ptr!();
*__slate_slot_3 = __ptr as *mut ();
*__slate_slot_2 = __x;
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") *__slate_slot_3 as u64 => __t0, in("ax") *__slate_slot_2 as u32, options(att_syntax));
*__slate_slot_4 = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
instrument_put_user!(__x, __ptr);
return *__slate_slot_4 as i64;
} else {
if __t26 == 3225948787 {
return blk_trace_setup((*(*sdp).device).request_queue, (*sdp).name.as_mut_ptr() as *mut u8, MKDEV!(21i32, unsafe { (*sdp).index }), core::ptr::null_mut::<block_device>(), p as *mut u8) as i64;
} else {
if __t26 == 4724 {
return blk_trace_startstop((*(*sdp).device).request_queue, 1i32) as i64;
} else {
if __t26 == 4725 {
return blk_trace_startstop((*(*sdp).device).request_queue, 0i32) as i64;
} else {
if __t26 == 4726 {
return blk_trace_remove((*(*sdp).device).request_queue) as i64;
} else {
if __t26 != 21378 {
if __t26 != 21382 {
if __t26 != 21381 {
if __t26 != 8709 {
if __t26 != 8836 {
if *__slate_slot_128 != 0 {
return 1i32.wrapping_neg() as i64;
} else {
// don't know so take safe approach
break '__join_2;
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
}
}
}
}
if atomic_read(core::ptr::addr_of_mut!((*sdp).detaching) as *const __SlateRecord_types_h_190) != 0 {
return 19i32.wrapping_neg() as i64;
}
}
*__slate_slot_130 = scsi_ioctl_block_when_processing_errors((*sdp).device, cmd_in as i32, (*filp).f_flags & 2048u32 != 0);
if *__slate_slot_130 != 0 {
return *__slate_slot_130 as i64;
} else {
return 515i32.wrapping_neg() as i64;
}
}
// -ERESTARTSYS because signal hit process
// N.B. User receives timeout as return value
// strange ..., for backward compatibility
// N.B. This ioctl never worked properly, but failed to
// return an error value. So returning '0' to keep compability
// with legacy applications.
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_ioctl")]
extern "C" fn sg_ioctl(filp: *mut file, cmd_in: u32, arg: u64) -> i64 {
let p: *mut () = arg as *mut ();
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
let __v0: *mut sg_fd = (unsafe { (*filp).private_data }) as *mut sg_fd;
let sfp: *mut sg_fd = __v0;
let __v1: bool = if __v0.is_null() {
true
} else {
let __v2: *mut sg_device = unsafe { (*sfp).parentdp };
sdp = __v2;
__v2.is_null()
};
if __v1 {
return 6i32.wrapping_neg() as i64;
}
ret = sg_ioctl_common(filp, sdp, sfp, cmd_in, p) as i32;
if ret != 515i32.wrapping_neg() {
return ret as i64;
}
(unsafe { scsi_ioctl((*sdp).device, (*filp).f_mode & 2u32 != 0, cmd_in as i32, p) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_poll")]
extern "C" fn sg_poll(filp: *mut file, wait: *mut poll_table_struct) -> u32 {
let mut res: u32 = 0;
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let mut srp: *mut sg_request = unsafe { core::mem::zeroed() };
let mut count: i32 = 0;
let mut iflags: u64 = 0;
sfp = (unsafe { (*filp).private_data }) as *mut sg_fd;
if sfp.is_null() {
return 8;
}
sdp = unsafe { (*sfp).parentdp };
if sdp.is_null() {
return 8;
}
poll_wait(filp, unsafe { core::ptr::addr_of_mut!((*sfp).read_wait) }, wait);
read_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
srp = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) });
while list_entry_is_head!(srp, unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) }, entry) == 0 {
// if any read waiting, flag it
if 0 == res && 1 == ((unsafe { (*srp).done }) as i32) && (unsafe { (*srp).sg_io_owned }) == 0 {
res = 1 | 64;
}
count = count.wrapping_add(1);
srp = list_next_entry__3!(srp, entry);
}
read_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
if atomic_read((unsafe { core::ptr::addr_of_mut!((*sdp).detaching) }) as *const __SlateRecord_types_h_190) != 0 {
res |= 16;
} else {
if (unsafe { (*sfp).cmd_q }) == 0 {
if 0 == count {
res |= 4 | 256;
}
} else {
if count < 16 {
res |= 4 | 256;
}
}
}
SCSI_LOG_TIMEOUT!();
res
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_fasync")]
extern "C" fn sg_fasync(fd: i32, filp: *mut file, mode: i32) -> i32 {
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let __v0: *mut sg_fd = (unsafe { (*filp).private_data }) as *mut sg_fd;
sfp = __v0;
let __v1: bool = if __v0.is_null() {
true
} else {
let __v2: *mut sg_device = unsafe { (*sfp).parentdp };
__v2.is_null()
};
if __v1 {
return 6i32.wrapping_neg();
}
SCSI_LOG_TIMEOUT!();
unsafe { fasync_helper(fd, filp, mode, core::ptr::addr_of_mut!((*sfp).async_qp)) }
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_vma_fault")]
extern "C" fn sg_vma_fault(vmf: *mut vm_fault) -> u32 {
let vma: *mut vm_area_struct = unsafe { (*vmf).__slate_anon_0.vma };
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let mut offset: u64 = 0;
let mut len: u64 = 0;
let mut sa: u64 = 0;
let mut k: i32 = 0;
let mut length: i32 = 0;
let __v0: bool = if vma.is_null() {
true
} else {
let __v1: *mut sg_fd = (unsafe { (*vma).vm_private_data }) as *mut sg_fd;
sfp = __v1;
__v1.is_null()
};
if __v0 {
return 2;
}
let rsv_schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*sfp).reserve) };
offset = (unsafe { (*vmf).__slate_anon_0.pgoff }) << 12i32;
if offset >= ((unsafe { (*rsv_schp).bufflen }) as u64) {
return 2;
}
SCSI_LOG_TIMEOUT!();
sa = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_start };
length = 1i32 << 12i32.wrapping_add(unsafe { (*rsv_schp).page_order });
while k < ((unsafe { (*rsv_schp).k_use_sg }) as i32) && sa < unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end } {
len = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_sub(sa);
len = if len < (length as u64) { len } else { length as u64 };
if offset < len {
let page: *mut page = unsafe { *unsafe { (*rsv_schp).pages }.wrapping_offset(k as isize) }.wrapping_offset((offset >> 12i32) as isize);
get_page(page); // increment page count
unsafe {
(*vmf).page = page;
}
return 0; // success
}
sa = sa.wrapping_add(len);
offset = offset.wrapping_sub(len);
k = k.wrapping_add(1);
}
2
}

static sg_mmap_vm_ops: __SlateConst<vm_operations_struct> = __SlateConst({
let mut __t0: vm_operations_struct = unsafe { core::mem::zeroed() };
__t0.fault = Some(sg_vma_fault);
__t0
});

#[unsafe(link_section = ".text.slate_distinct.sg.sg_mmap")]
extern "C" fn sg_mmap(filp: *mut file, vma: *mut vm_area_struct) -> i32 {
let mut __v5: bool = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut length: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(length) as *mut i32;
let mut k: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(k) as *mut i32;
let mut rsv_schp: core::mem::MaybeUninit<*mut sg_scatter_hold> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut sg_scatter_hold = core::ptr::addr_of_mut!(rsv_schp) as *mut *mut sg_scatter_hold;
let mut sa: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(sa) as *mut u64;
let mut len: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u64 = core::ptr::addr_of_mut!(len) as *mut u64;
let mut req_sz: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u64 = core::ptr::addr_of_mut!(req_sz) as *mut u64;
let mut sfp: core::mem::MaybeUninit<*mut sg_fd> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut sg_fd = core::ptr::addr_of_mut!(sfp) as *mut *mut sg_fd;
unsafe {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_5 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
ret = 0;
if filp.is_null() || vma.is_null() {
__v5 = true;
} else {
let __v4: *mut sg_fd = (*filp).private_data as *mut sg_fd;
*__slate_slot_13 = __v4;
__v5 = __v4.is_null();
}
if __v5 {
return 6i32.wrapping_neg();
} else {
*__slate_slot_12 = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_sub((*vma).__slate_anon_0.__slate_anon_0.vm_start);
SCSI_LOG_TIMEOUT!();
if (*vma).vm_pgoff != 0 {
return 22i32.wrapping_neg();
} else {
// want no offset
*__slate_slot_9 = core::ptr::addr_of_mut!((*(*__slate_slot_13)).reserve);
mutex_lock(core::ptr::addr_of_mut!((*(*__slate_slot_13)).f_mutex));
if *__slate_slot_12 > ((*(*__slate_slot_9)).bufflen as u64) {
ret = 12i32.wrapping_neg(); // cannot map more than reserved buffer
} else {
*__slate_slot_10 = (*vma).__slate_anon_0.__slate_anon_0.vm_start;
*__slate_slot_7 = 1i32 << 12i32.wrapping_add((*(*__slate_slot_9)).page_order);
*__slate_slot_8 = 0;
while *__slate_slot_8 < ((*(*__slate_slot_9)).k_use_sg as i32) && *__slate_slot_10 < (*vma).__slate_anon_0.__slate_anon_0.vm_end {
*__slate_slot_11 = unsafe { (*vma).__slate_anon_0.__slate_anon_0.vm_end }.wrapping_sub(*__slate_slot_10);
*__slate_slot_11 = if *__slate_slot_11 < (*__slate_slot_7 as u64) { *__slate_slot_11 } else { *__slate_slot_7 as u64 };
let __v3: u64 = *__slate_slot_10;
*__slate_slot_10 = __v3.wrapping_add(*__slate_slot_11);
let __v1: i32 = *__slate_slot_8;
*__slate_slot_8 = __v1.wrapping_add(1);
}
(*(*__slate_slot_13)).mmap_called = 1;
vm_flags_set(vma, BIT!(14i32) | BIT!(18i32) | BIT!(26i32));
(*vma).vm_private_data = *__slate_slot_13 as *mut ();
(*vma).vm_ops = core::ptr::addr_of!(sg_mmap_vm_ops.0);
}
mutex_unlock(core::ptr::addr_of_mut!((*(*__slate_slot_13)).f_mutex));
return ret;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_rq_end_io_usercontext")]
extern "C" fn sg_rq_end_io_usercontext(work: *mut work_struct) {
let mut __v0: *mut sg_request;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__8!(sg_request) as isize)) }) as *mut ()) as *mut sg_request;
let srp: *mut sg_request = __v0;
let sfp: *mut sg_fd = unsafe { (*srp).parentfp };
sg_finish_rem_req(srp);
sg_remove_request(sfp, srp);
kref_put(unsafe { core::ptr::addr_of_mut!((*sfp).f_ref) }, Some(sg_remove_sfp));
}

/// This function is a "bottom half" handler that is called by the mid
/// level when a command is completed (or has failed).
#[unsafe(link_section = ".text.slate_distinct.sg.sg_rq_end_io")]
extern "C" fn sg_rq_end_io(rq: *mut request, status: u8, iob: *const io_comp_batch) -> u32 {
let mut __slate_compound_19: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let scmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
let srp: *mut sg_request = (unsafe { (*rq).end_io_data }) as *mut sg_request;
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let mut iflags: u64 = 0;
let mut ms: u32 = 0;
let mut result: i32 = 0;
let mut resid: i32 = 0;
let mut done: i32 = 1;
let mut __v0: i64;
{
let __ret_warn_on: i32 = (((unsafe { (*srp).done }) as i32) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("605: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 605b - ., 3; .popsection\n# {0}", const 605 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_sg_rq_end_io_1, sym __slate_asm_string_sg_rq_end_io_2, const 1324 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("606: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 606b - ., 4; .popsection\n# {0}", const 606 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 != 0 {
return 0;
}
sfp = unsafe { (*srp).parentfp };
let mut __v5: i64;
let __ret_warn_on: i32 = sfp.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("607: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 607b - ., 3; .popsection\n# {0}", const 607 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_sg_rq_end_io_6, sym __slate_asm_string_sg_rq_end_io_7, const 1328 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("608: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 608b - ., 4; .popsection\n# {0}", const 608 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return 0;
}
sdp = unsafe { (*sfp).parentdp };
if unlikely__2!(atomic_read((unsafe { core::ptr::addr_of_mut!((*sdp).detaching) }) as *const __SlateRecord_types_h_190)) != 0 {
let mut __v10: i32;
while false {
}
__v10 = unsafe { _printk(c"\x016%s: device detaching\n".as_ptr() as *const u8, c"sg_rq_end_io".as_ptr() as *mut u8) };
}
let sense: *mut u8 = (unsafe { (*scmd).sense_buffer }) as *mut u8;
result = unsafe { (*scmd).result };
resid = (unsafe { (*scmd).resid_len }) as i32;
SCSI_LOG_TIMEOUT!();
unsafe {
(*srp).header.resid = resid;
}
if 0 != result {
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
unsafe {
(*srp).header.status = (255 & result) as u8;
}
unsafe {
(*srp).header.masked_status = sg_status_byte!(result) as u8;
}
unsafe {
(*srp).header.msg_status = 0;
}
unsafe {
(*srp).header.host_status = host_byte!(result) as u16;
}
unsafe {
(*srp).header.driver_status = driver_byte!(result) as u16;
}
if ((unsafe { (*sdp).sgdebug }) as i32) > 0 && (1 == ((unsafe { (*srp).header.masked_status }) as i32) || 17 == ((unsafe { (*srp).header.masked_status }) as i32)) {
unsafe { __scsi_print_sense((*sdp).device as *const scsi_device, c"sg_rq_end_io".as_ptr() as *const u8, sense as *const u8, 96i32) };
}
// Following if statement is a patch supplied by Eric Youngdale
let __v13: bool = driver_byte!(result) != 0 && unsafe { scsi_normalize_sense(sense as *const u8, 96i32, core::ptr::addr_of_mut!(sshdr)) };
let __v14: bool = __v13 && !scsi_sense_is_deferred(unsafe { &(*(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr)) });
if __v14 && (sshdr.sense_key as i32) == 6 && ((unsafe { (*(*sdp).device).__slate_bits_0.__get_removable() }) as i32) != 0 {
// Detected possible disc change. Set the bit - this
//
// may be used if there are filesystems using this device
unsafe {
(*(*sdp).device).__slate_bits_0.__set_changed(1u32);
}
}
}
if (unsafe { (*scmd).sense_len }) != 0 {
unsafe { memcpy(((*srp).sense_b.as_mut_ptr() as *mut u8) as *mut (), (*scmd).sense_buffer as *const (), 96u64) };
}
// Rely on write phase to clean out srp status values, so no "else"
// Free the request as soon as it is complete so that its resources
// can be reused without waiting for userspace to read() the
// result.  But keep the associated bio (if any) around until
// blk_rq_unmap_user() can be called from user context.
unsafe {
(*srp).rq = core::ptr::null_mut::<request>();
}
unsafe { blk_mq_free_request(rq) };
write_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
if unlikely__5!(unsafe { (*srp).orphan }) != 0 {
if (unsafe { (*sfp).keep_orphan }) != 0 {
unsafe {
(*srp).sg_io_owned = 0;
}
} else {
done = 0;
}
}
unsafe {
(*srp).done = done as u8;
}
ms = jiffies_to_msecs(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) });
unsafe {
(*srp).header.duration = if ms > (*srp).header.duration { ms.wrapping_sub((*srp).header.duration) } else { 0 };
}
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
if likely__2!(done) != 0 {
// Now wake up any sg_read() that is waiting for this
// packet.
wake_up_interruptible!(unsafe { core::ptr::addr_of_mut!((*sfp).read_wait) });
unsafe { kill_fasync(core::ptr::addr_of_mut!((*sfp).async_qp), 29i32, 1i32) };
kref_put(unsafe { core::ptr::addr_of_mut!((*sfp).f_ref) }, Some(sg_remove_sfp));
} else {
'__slate_break_21: loop {
'__slate_break_20: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*srp).ew.work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*srp).ew.work)).data = *{
__slate_compound_19 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_19)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*srp).ew.work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*srp).ew.work)).func = Some(sg_rq_end_io_usercontext);
}
if true {
break;
}
}
if true {
break;
}
}
schedule_work(unsafe { core::ptr::addr_of_mut!((*srp).ew.work) });
}
0
}

static mut __key__1404: lock_class_key = unsafe { core::mem::zeroed() };

static sg_fops: __SlateConst<file_operations> = __SlateConst({
let mut __t0: file_operations = unsafe { core::mem::zeroed() };
__t0.owner = core::ptr::null_mut::<module>();
__t0.read = Some(sg_read);
__t0.write = Some(sg_write);
__t0.poll = Some(sg_poll);
__t0.unlocked_ioctl = Some(sg_ioctl);
__t0.compat_ioctl = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, u32, u64) -> i64>>(compat_ptr_ioctl as *const ()) };
__t0.mmap = Some(sg_mmap);
__t0.open = Some(sg_open);
__t0.release = Some(sg_release);
__t0.fasync = Some(sg_fasync);
__t0
});

static sg_sysfs_class: __SlateConst<class> = __SlateConst({
let mut __t0: class = unsafe { core::mem::zeroed() };
__t0.name = (b"scsi_generic\0".as_ptr() as *mut u8) as *const u8;
__t0
});

static mut sg_sysfs_valid: i32 = 0;

fn sg_alloc(scsidp: *mut scsi_device) -> *mut sg_device {
let mut __slate_compound_13: rwlock = unsafe { core::mem::zeroed() };
let mut __slate_compound_17: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut k: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u32 = core::ptr::addr_of_mut!(k) as *mut u32;
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut iflags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u64 = core::ptr::addr_of_mut!(iflags) as *mut u64;
let mut sdp: core::mem::MaybeUninit<*mut sg_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut sg_device = core::ptr::addr_of_mut!(sdp) as *mut *mut sg_device;
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
unsafe {
q = (*scsidp).request_queue;
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
__obj_size = size_mul(core::mem::size_of::<sg_device>() as u64, 1u64);
let __slate_auto_bytes_5 = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_0 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_17 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_17)
});
alloc_tag_restore!();
} else {
*__slate_slot_1 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
}
*__slate_slot_9 = *__slate_slot_1 as *mut sg_device;
if (*__slate_slot_9).is_null() {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, scsidp as *const scsi_device, core::ptr::null::<u8>(), c"%s: kmalloc Sg_device failure\n".as_ptr() as *const u8, c"sg_alloc".as_ptr() as *mut u8);
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut sg_device;
} else {
idr_preload(((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
write_lock_irqsave!(core::ptr::addr_of_mut!(sg_index_lock), *__slate_slot_8);
*__slate_slot_7 = idr_alloc(core::ptr::addr_of_mut!(sg_index_idr), *__slate_slot_9 as *mut (), 0i32, 1048576i32, (BIT!(11i32) as u32) | (BIT!(13i32) as u32));
if *__slate_slot_7 < 0 {
if *__slate_slot_7 == 28i32.wrapping_neg() {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, scsidp as *const scsi_device, core::ptr::null::<u8>(), c"Unable to attach sg device type=%d, minor number exceeds %d\n".as_ptr() as *const u8, (*scsidp).r#type as i32, 1048575i32);
*__slate_slot_7 = 19i32.wrapping_neg();
} else {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, scsidp as *const scsi_device, core::ptr::null::<u8>(), c"%s: idr allocation Sg_device failure: %d\n".as_ptr() as *const u8, c"sg_alloc".as_ptr() as *mut u8, *__slate_slot_7);
}
} else {
*__slate_slot_6 = *__slate_slot_7 as u32;
SCSI_LOG_TIMEOUT!();
sprintf((*(*__slate_slot_9)).name.as_mut_ptr() as *mut u8, c"sg%d".as_ptr() as *const u8, *__slate_slot_6);
(*(*__slate_slot_9)).device = scsidp;
loop {
__mutex_init(core::ptr::addr_of_mut!((*(*__slate_slot_9)).open_rel_lock), c"&sdp->open_rel_lock".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1468));
if true {
break;
}
}
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_9)).sfds));
loop {
__init_waitqueue_head(core::ptr::addr_of_mut!((*(*__slate_slot_9)).open_wait), c"&sdp->open_wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__1470));
if true {
break;
}
}
atomic_set(core::ptr::addr_of_mut!((*(*__slate_slot_9)).detaching), 0i32);
loop {
*core::ptr::addr_of_mut!((*(*__slate_slot_9)).sfd_lock) = *{
__slate_compound_13 = rwlock { raw_lock: qrwlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qrwlock_types_h_27 = core::mem::zeroed();
__t0.cnts = __SlateRecord_types_h_190 { counter: 0 };
__t0
}, wait_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } } };
core::ptr::addr_of_mut!(__slate_compound_13)
};
if true {
break;
}
}
(*(*__slate_slot_9)).sg_tablesize = queue_max_segments(q as *const request_queue) as i32;
(*(*__slate_slot_9)).index = *__slate_slot_6;
kref_init(core::ptr::addr_of_mut!((*(*__slate_slot_9)).d_ref));
*__slate_slot_7 = 0;
}
write_unlock_irqrestore!(core::ptr::addr_of_mut!(sg_index_lock), *__slate_slot_8);
idr_preload_end();
if *__slate_slot_7 != 0 {
kfree(*__slate_slot_9 as *const ());
return ERR_PTR(*__slate_slot_7 as i64) as *mut sg_device;
} else {
return *__slate_slot_9;
}
}
}
}

static mut __key__1468: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__1470: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.sg.sg_add_device")]
extern "C" fn sg_add_device(cl_dev: *mut device) -> i32 {
let mut sg_class_member: core::mem::MaybeUninit<*mut device> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut device = core::ptr::addr_of_mut!(sg_class_member) as *mut *mut device;
let mut iflags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(iflags) as *mut u64;
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut cdev: *mut cdev = unsafe { core::mem::zeroed() };
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut scsidp: *mut scsi_device = unsafe { core::mem::zeroed() };
unsafe {
scsidp = to_scsi_device!(unsafe { (*cl_dev).parent });
sdp = core::ptr::null_mut::<sg_device>();
cdev = core::ptr::null_mut::<cdev>();
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if !blk_get_queue((*scsidp).request_queue) {
while false {
}
_printk(c"\x014%s: get scsi_device queue failed\n".as_ptr() as *const u8, c"sg_add_device".as_ptr() as *mut u8);
return 19i32.wrapping_neg();
} else {
*__slate_slot_8 = 12i32.wrapping_neg();
cdev = cdev_alloc();
if cdev.is_null() {
while false {
}
_printk(c"\x014%s: cdev_alloc failed\n".as_ptr() as *const u8, c"sg_add_device".as_ptr() as *mut u8);
} else {
(*cdev).owner = core::ptr::null_mut::<module>();
(*cdev).ops = core::ptr::addr_of!(sg_fops.0);
sdp = sg_alloc(scsidp);
if IS_ERR(sdp as *const ()) {
while false {
}
_printk(c"\x014%s: sg_alloc failed\n".as_ptr() as *const u8, c"sg_add_device".as_ptr() as *mut u8);
*__slate_slot_8 = PTR_ERR(sdp as *const ()) as i32;
} else {
'__join_3: {
*__slate_slot_8 = cdev_add(cdev, MKDEV!(21i32, unsafe { (*sdp).index }), 1u32);
if *__slate_slot_8 == 0 {
(*sdp).cdev = cdev;
if sg_sysfs_valid != 0 {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_2 = device_create(core::ptr::addr_of!(sg_sysfs_class.0), (*cl_dev).parent, MKDEV!(21i32, unsafe { (*sdp).index }), sdp as *mut (), c"%s".as_ptr() as *const u8, (*sdp).name.as_mut_ptr() as *mut u8);
if IS_ERR(*__slate_slot_2 as *const ()) {
while false {
}
_printk(c"\x013%s: device_create failed\n".as_ptr() as *const u8, c"sg_add_device".as_ptr() as *mut u8);
*__slate_slot_8 = PTR_ERR(*__slate_slot_2 as *const ()) as i32;
break '__join_3;
} else {
*__slate_slot_8 = sysfs_create_link(core::ptr::addr_of_mut!((*scsidp).sdev_gendev.kobj), core::ptr::addr_of_mut!((*(*__slate_slot_2)).kobj), c"generic".as_ptr() as *const u8);
if *__slate_slot_8 != 0 {
while false {
}
_printk(c"\x013%s: unable to make symlink 'generic' back to sg%d\n".as_ptr() as *const u8, c"sg_add_device".as_ptr() as *mut u8, (*sdp).index);
}
}
} else {
while false {
}
_printk(c"\x014%s: sg_sys Invalid\n".as_ptr() as *const u8, c"sg_add_device".as_ptr() as *mut u8);
}
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, scsidp as *const scsi_device, core::ptr::null::<u8>(), c"Attached scsi generic sg%d type %d\n".as_ptr() as *const u8, (*sdp).index, (*scsidp).r#type as i32);
dev_set_drvdata(cl_dev, sdp as *mut ());
return 0;
}
}
write_lock_irqsave!(core::ptr::addr_of_mut!(sg_index_lock), *__slate_slot_7);
idr_remove(core::ptr::addr_of_mut!(sg_index_idr), (*sdp).index as u64);
write_unlock_irqrestore!(core::ptr::addr_of_mut!(sg_index_lock), *__slate_slot_7);
kfree(sdp as *const ());
}
}
if !cdev.is_null() {
cdev_del(cdev);
}
blk_put_queue((*scsidp).request_queue);
return *__slate_slot_8;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_device_destroy")]
extern "C" fn sg_device_destroy(kref: *mut kref) {
let sdp: *mut sg_device = container_of__23!(kref, sg_device);
let q: *mut request_queue = unsafe { (*(*sdp).device).request_queue };
let mut flags: u64 = 0;
// CAUTION!  Note that the device can still be found via idr_find()
// even though the refcount is 0.  Therefore, do idr_remove() BEFORE
// any other cleanup.
unsafe { blk_trace_remove(q) };
unsafe { blk_put_queue(q) };
write_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, flags);
unsafe { idr_remove(core::ptr::addr_of_mut!(sg_index_idr), (*sdp).index as u64) };
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, flags);
SCSI_LOG_TIMEOUT!();
unsafe { kfree(sdp as *const ()) };
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_remove_device")]
extern "C" fn sg_remove_device(cl_dev: *mut device) {
let scsidp: *mut scsi_device = to_scsi_device!(unsafe { (*cl_dev).parent });
let sdp: *mut sg_device = dev_get_drvdata(cl_dev as *const device) as *mut sg_device;
let mut iflags: u64 = 0;
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let mut val: i32 = 0;
if sdp.is_null() {
return;
}
// want sdp->detaching non-zero as soon as possible
val = atomic_inc_return(unsafe { core::ptr::addr_of_mut!((*sdp).detaching) });
if val > 1 {
return;
}
// only want to do following once per device
SCSI_LOG_TIMEOUT!();
read_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sdp).sfd_lock) }, iflags);
sfp = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!((*sdp).sfds) });
while list_entry_is_head!(sfp, unsafe { core::ptr::addr_of_mut!((*sdp).sfds) }, sfd_siblings) == 0 {
wake_up_interruptible_all!(unsafe { core::ptr::addr_of_mut!((*sfp).read_wait) });
unsafe { kill_fasync(core::ptr::addr_of_mut!((*sfp).async_qp), 29i32, 6i32) };
sfp = list_next_entry__5!(sfp, sfd_siblings);
}
wake_up_interruptible_all!(unsafe { core::ptr::addr_of_mut!((*sdp).open_wait) });
read_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sdp).sfd_lock) }, iflags);
unsafe { sysfs_remove_link(core::ptr::addr_of_mut!((*scsidp).sdev_gendev.kobj), c"generic".as_ptr() as *const u8) };
unsafe { device_destroy(core::ptr::addr_of!(sg_sysfs_class.0), MKDEV!(21i32, unsafe { (*sdp).index })) };
unsafe { cdev_del((*sdp).cdev) };
unsafe {
(*sdp).cdev = core::ptr::null_mut::<cdev>();
}
kref_put(unsafe { core::ptr::addr_of_mut!((*sdp).d_ref) }, Some(sg_device_destroy));
}

fn __check_scatter_elem_sz() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(scatter_elem_sz) }
}

static __param_str_scatter_elem_sz: __SlateAlign16<[u8; 19]> = __SlateAlign16([115u8, 103u8, 46u8, 115u8, 99u8, 97u8, 116u8, 116u8, 101u8, 114u8, 95u8, 101u8, 108u8, 101u8, 109u8, 95u8, 115u8, 122u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_scatter_elem_sz: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_scatter_elem_sz.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_sg_c_1624>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1624_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1624_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1624_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1624_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1624_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(scatter_elem_sz) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_609: [u8; 32] = [115, 103, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 115, 99, 97, 116, 116, 101, 114, 95, 101, 108, 101, 109, 95, 115, 122, 58, 105, 110, 116, 0];

fn __check_allow_dio() -> *mut i32 {
unsafe { core::ptr::addr_of_mut!(sg_allow_dio) }
}

static __param_str_allow_dio: [u8; 13] = [115, 103, 46, 97, 108, 108, 111, 119, 95, 100, 105, 111, 0];

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_allow_dio: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_allow_dio) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(param_ops_int) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_sg_c_1625>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1625_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1625_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1625_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1625_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1625_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(sg_allow_dio) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_610: [u8; 26] = [115, 103, 46, 112, 97, 114, 109, 116, 121, 112, 101, 61, 97, 108, 108, 111, 119, 95, 100, 105, 111, 58, 105, 110, 116, 0];

#[unsafe(link_section = ".text.slate_distinct.sg.def_reserved_size_set")]
extern "C" fn def_reserved_size_set(val: *const u8, kp: *const kernel_param) -> i32 {
let mut size: i32 = 0;
let mut ret: i32 = 0;
if val.is_null() {
return 22i32.wrapping_neg();
}
ret = unsafe { kstrtoint(val, 0u32, core::ptr::addr_of_mut!(size)) };
if ret != 0 {
return ret;
}
// limit to 1 MB
if size < 0 || size > 1048576 {
return 34i32.wrapping_neg();
}
unsafe {
def_reserved_size = size;
}
0
}

static def_reserved_size_ops: __SlateConst<kernel_param_ops> = __SlateConst({
let mut __t0: kernel_param_ops = unsafe { core::mem::zeroed() };
__t0.set = Some(def_reserved_size_set);
__t0.get = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut u8, *const kernel_param) -> i32>>(param_get_int as *const ()) };
__t0
});

static __param_str_def_reserved_size: __SlateAlign16<[u8; 21]> = __SlateAlign16([115u8, 103u8, 46u8, 100u8, 101u8, 102u8, 95u8, 114u8, 101u8, 115u8, 101u8, 114u8, 118u8, 101u8, 100u8, 95u8, 115u8, 105u8, 122u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = "__param")]
static __param_def_reserved_size: __SlateConst<kernel_param> = __SlateConst({
let mut __t0: kernel_param = unsafe { core::mem::zeroed() };
__t0.name = unsafe { core::ptr::addr_of!(__param_str_def_reserved_size.0) as *const u8 };
__t0.r#mod = core::ptr::null_mut::<module>();
__t0.ops = unsafe { core::ptr::addr_of!(def_reserved_size_ops.0) };
__t0.perm = ((core::mem::size_of::<__SlateRecord_sg_c_1651>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1651_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1651_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1651_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1651_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sg_c_1651_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0.level = 1i32.wrapping_neg() as i8;
__t0.flags = 0u8;
__t0.__slate_anon_6 = __SlateRecord_moduleparam_h_89 { arg: (unsafe { core::ptr::addr_of_mut!(def_reserved_size) }) as *mut () };
__t0
});

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_611: [u8; 26] = [115, 103, 46, 97, 117, 116, 104, 111, 114, 61, 68, 111, 117, 103, 108, 97, 115, 32, 71, 105, 108, 98, 101, 114, 116, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_612: [u8; 40] = [115, 103, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 83, 67, 83, 73, 32, 103, 101, 110, 101, 114, 105, 99, 32, 40, 115, 103, 41, 32, 100, 114, 105, 118, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_613: [u8; 24] = [115, 103, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 115, 99, 115, 105, 47, 115, 103, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_614: [u8; 15] = [115, 103, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_615: [u8; 18] = [115, 103, 46, 118, 101, 114, 115, 105, 111, 110, 61, 51, 46, 53, 46, 51, 54, 0];

#[used(compiler)]
#[unsafe(link_section = "__modver")]
static __modver_attr: __SlateConst<module_version_attribute> = __SlateConst(module_version_attribute { mattr: {
let mut __t0: module_attribute = unsafe { core::mem::zeroed() };
__t0.attr = {
let mut __t1: attribute = unsafe { core::mem::zeroed() };
__t1.name = (b"version\0".as_ptr() as *mut u8) as *const u8;
__t1.mode = 292u16;
__t1
};
__t0.show = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*const module_attribute, *mut module_kobject, *mut u8) -> i64>>(__modver_version_show as *const ()) };
__t0
}, module_name: (b"sg\0".as_ptr() as *mut u8) as *const u8, version: (b"3.5.36\0".as_ptr() as *mut u8) as *const u8 });

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_616: [u8; 25] = [115, 103, 46, 97, 108, 105, 97, 115, 61, 99, 104, 97, 114, 45, 109, 97, 106, 111, 114, 45, 50, 49, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_617: [u8; 93] = [115, 103, 46, 112, 97, 114, 109, 61, 115, 99, 97, 116, 116, 101, 114, 95, 101, 108, 101, 109, 95, 115, 122, 58, 115, 99, 97, 116, 116, 101, 114, 32, 103, 97, 116, 104, 101, 114, 32, 101, 108, 101, 109, 101, 110, 116, 32, 115, 105, 122, 101, 32, 40, 100, 101, 102, 97, 117, 108, 116, 58, 32, 109, 97, 120, 40, 83, 71, 95, 83, 67, 65, 84, 84, 69, 82, 95, 83, 90, 44, 32, 80, 65, 71, 69, 95, 83, 73, 90, 69, 41, 41, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_618: [u8; 62] = [115, 103, 46, 112, 97, 114, 109, 61, 100, 101, 102, 95, 114, 101, 115, 101, 114, 118, 101, 100, 95, 115, 105, 122, 101, 58, 115, 105, 122, 101, 32, 111, 102, 32, 98, 117, 102, 102, 101, 114, 32, 114, 101, 115, 101, 114, 118, 101, 100, 32, 102, 111, 114, 32, 101, 97, 99, 104, 32, 102, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_619: [u8; 59] = [115, 103, 46, 112, 97, 114, 109, 61, 97, 108, 108, 111, 119, 95, 100, 105, 111, 58, 97, 108, 108, 111, 119, 32, 100, 105, 114, 101, 99, 116, 32, 73, 47, 79, 32, 40, 100, 101, 102, 97, 117, 108, 116, 58, 32, 48, 32, 40, 100, 105, 115, 97, 108, 108, 111, 119, 41, 41, 0];

#[unsafe(link_section = ".init.text")]
extern "C" fn init_sg() -> i32 {
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
unsafe {
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (scatter_elem_sz as u64) < _AC!() << 12i32 {
scatter_elem_sz = (_AC!() << 12i32) as i32;
scatter_elem_sz_prev = scatter_elem_sz;
}
*__slate_slot_0 = register_chrdev_region(MKDEV__2!(21i32, 0i32) as u32, 1048576u32, c"sg".as_ptr() as *const u8);
if *__slate_slot_0 != 0 {
return *__slate_slot_0;
} else {
*__slate_slot_0 = class_register(core::ptr::addr_of!(sg_sysfs_class.0));
if *__slate_slot_0 == 0 {
sg_sysfs_valid = 1;
*__slate_slot_0 = scsi_register_interface(core::ptr::addr_of_mut!(sg_interface));
if 0 == *__slate_slot_0 {
sg_proc_init();
return 0;
} else {
class_unregister(core::ptr::addr_of!(sg_sysfs_class.0));
}
}
unregister_chrdev_region(MKDEV__2!(21i32, 0i32) as u32, 1048576u32);
return *__slate_slot_0;
}
}
}

#[unsafe(link_section = ".exit.text")]
extern "C" fn exit_sg() {
unsafe { remove_proc_subtree(c"scsi/sg".as_ptr() as *const u8, core::ptr::null_mut::<proc_dir_entry>()) };
scsi_unregister_interface!(unsafe { core::ptr::addr_of_mut!(sg_interface) });
unsafe { class_unregister(core::ptr::addr_of!(sg_sysfs_class.0)) };
unsafe {
sg_sysfs_valid = 0;
}
unsafe { unregister_chrdev_region(MKDEV__2!(21i32, 0i32) as u32, 1048576u32) };
unsafe { idr_destroy(core::ptr::addr_of_mut!(sg_index_idr)) };
}

fn sg_start_req(srp: *mut sg_request, cmd: *mut u8) -> i32 {
let mut res: i32 = 0;
let sfp: *mut sg_fd = unsafe { (*srp).parentfp };
let hp: *mut sg_io_hdr = unsafe { core::ptr::addr_of_mut!((*srp).header) };
let dxfer_len: i32 = (unsafe { (*hp).dxfer_len }) as i32;
let dxfer_dir: i32 = unsafe { (*hp).dxfer_direction };
let iov_count: u32 = (unsafe { (*hp).iovec_count }) as u32;
let req_schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*srp).data) };
let rsv_schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*sfp).reserve) };
let q: *mut request_queue = unsafe { (*(*(*sfp).parentdp).device).request_queue };
let mut md: *mut rq_map_data = unsafe { core::mem::zeroed() };
let mut map_data: rq_map_data = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(map_data) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 24usize);
}
let rw: i32 = if (unsafe { (*hp).dxfer_direction }) == 2i32.wrapping_neg() { 1 } else { 0 };
SCSI_LOG_TIMEOUT!();
// NOTE
//
// With scsi-mq enabled, there are a fixed number of preallocated
// requests equal in number to shost->can_queue.  If all of the
// preallocated requests are already in use, then scsi_alloc_request()
// will sleep until an active command completes, freeing up a request.
// Although waiting in an asynchronous interface is less than ideal, we
// do not want to use BLK_MQ_REQ_NOWAIT here because userspace might
// not expect an EWOULDBLOCK from this condition.
let rq: *mut request = unsafe { scsi_alloc_request(q, (if (*hp).dxfer_direction == 2i32.wrapping_neg() { 35i32 } else { 34i32 }) as u32, 0u32) };
if IS_ERR(rq as *const ()) {
return PTR_ERR(rq as *const ()) as i32;
}
let scmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
if (((unsafe { (*hp).cmd_len }) as i32) as u64) > (core::mem::size_of::<[u8; 32]>() as u64) {
unsafe { blk_mq_free_request(rq) };
return 22i32.wrapping_neg();
}
unsafe { memcpy(((*scmd).cmnd.as_mut_ptr() as *mut u8) as *mut (), cmd as *const (), (*hp).cmd_len as u64) };
unsafe {
(*scmd).cmd_len = (*hp).cmd_len as u16;
}
unsafe {
(*srp).rq = rq;
}
unsafe {
(*rq).end_io_data = srp as *mut ();
}
unsafe {
(*scmd).allowed = 0;
}
if dxfer_len <= 0 || dxfer_dir == 1i32.wrapping_neg() {
return 0;
}
let __v1: bool = (unsafe { sg_allow_dio }) != 0 && (unsafe { (*hp).flags }) & 1 != 0 && dxfer_dir != 5i32.wrapping_neg() && iov_count == 0 && blk_rq_aligned(q, (unsafe { (*hp).dxferp }) as u64, dxfer_len as u32);
if __v1 {
md = core::ptr::null_mut::<rq_map_data>();
} else {
md = core::ptr::addr_of_mut!(map_data);
}
if !md.is_null() {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sfp).f_mutex)) };
if (dxfer_len as u32) <= unsafe { (*rsv_schp).bufflen } && (unsafe { (*sfp).res_in_use }) == 0 {
unsafe {
(*sfp).res_in_use = 1;
}
sg_link_reserve(sfp, srp, dxfer_len);
} else {
if (unsafe { (*hp).flags }) & 4 != 0 {
res = 16i32.wrapping_neg(); // sfp->res_in_use == 1
if (dxfer_len as u32) > unsafe { (*rsv_schp).bufflen } {
res = 12i32.wrapping_neg();
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sfp).f_mutex)) };
return res;
} else {
res = sg_build_indirect(req_schp, sfp, dxfer_len);
if res != 0 {
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sfp).f_mutex)) };
return res;
}
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sfp).f_mutex)) };
unsafe {
(*md).pages = (*req_schp).pages;
}
unsafe {
(*md).page_order = (*req_schp).page_order as u16;
}
unsafe {
(*md).nr_entries = (*req_schp).k_use_sg;
}
unsafe {
(*md).offset = 0;
}
unsafe {
(*md).null_mapped = unsafe { (*hp).dxferp }.is_null();
}
if dxfer_dir == 4i32.wrapping_neg() {
unsafe {
(*md).from_user = true;
}
} else {
unsafe {
(*md).from_user = false;
}
}
}
res = unsafe { blk_rq_map_user_io(rq, md, (*hp).dxferp, (*hp).dxfer_len as u64, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32), iov_count != 0, iov_count as i32, true, rw) };
if res == 0 {
unsafe {
(*srp).bio = (*rq).bio;
}
if md.is_null() {
unsafe {
(*req_schp).dio_in_use = 1;
}
unsafe {
(*hp).info = (*hp).info | 2;
}
}
}
res
}

fn sg_finish_rem_req(srp: *mut sg_request) -> i32 {
let mut ret: i32 = 0;
let sfp: *mut sg_fd = unsafe { (*srp).parentfp };
let req_schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*srp).data) };
SCSI_LOG_TIMEOUT!();
if !unsafe { (*srp).bio }.is_null() {
ret = unsafe { blk_rq_unmap_user((*srp).bio) };
}
if !unsafe { (*srp).rq }.is_null() {
unsafe { blk_mq_free_request((*srp).rq) };
}
if (unsafe { (*srp).res_used }) != 0 {
sg_unlink_reserve(sfp, srp);
} else {
sg_remove_scat(sfp, req_schp);
}
ret
}

fn sg_build_sgat(schp: *mut sg_scatter_hold, sfp: *const sg_fd, tablesize: i32) -> i32 {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let sg_bufflen: i32 = (tablesize as u64).wrapping_mul(core::mem::size_of::<*mut page>() as u64) as i32;
let gfp_flags: u32 = (BIT!(5i32) as u32) | (BIT!(11i32) as u32) | (BIT!(13i32) as u32);
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(sg_bufflen as u64, gfp_flags, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(sg_bufflen as u64, gfp_flags, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
unsafe {
(*schp).pages = __v0 as *mut *mut page;
}
if unsafe { (*schp).pages }.is_null() {
return 12i32.wrapping_neg();
}
unsafe {
(*schp).sglist_len = sg_bufflen as u32;
}
return tablesize; // number of scat_gath elements allocated
}

fn sg_build_indirect(schp: *mut sg_scatter_hold, sfp: *mut sg_fd, buff_size: i32) -> i32 {
let mut gfp_mask: u32 = unsafe { core::mem::zeroed() };
let mut order: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(order) as *mut i32;
let mut blk_size: i32 = unsafe { core::mem::zeroed() };
let mut sg_tablesize: i32 = unsafe { core::mem::zeroed() };
let mut mx_sc_elems: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(mx_sc_elems) as *mut i32;
let mut num: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(num) as *mut i32;
let mut rem_sz: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(rem_sz) as *mut i32;
let mut k: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(k) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut ret_sz: i32 = unsafe { core::mem::zeroed() };
unsafe {
ret_sz = 0;
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
sg_tablesize = (*(*sfp).parentdp).sg_tablesize;
blk_size = buff_size;
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
gfp_mask = (BIT!(5i32) as u32) | (BIT!(11i32) as u32) | (BIT!(18i32) as u32) | (BIT!(13i32) as u32) | (BIT!(8i32) as u32);
if blk_size < 0 {
return 14i32.wrapping_neg();
} else {
if 0 == blk_size {
blk_size = blk_size.wrapping_add(1);
}
// don't know why
// round request up to next highest SG_SECTOR_SZ byte boundary
blk_size = ALIGN__2!(blk_size, 512i32);
SCSI_LOG_TIMEOUT!();
// N.B. ret_sz carried into this block ...
*__slate_slot_15 = sg_build_sgat(schp, sfp as *const sg_fd, sg_tablesize);
if *__slate_slot_15 < 0 {
return *__slate_slot_15;
} else {
// most likely -ENOMEM
*__slate_slot_16 = scatter_elem_sz;
if unlikely!(*__slate_slot_16 != scatter_elem_sz_prev) != 0 {
if (*__slate_slot_16 as u64) < _AC!() << 12i32 {
scatter_elem_sz = (_AC!() << 12i32) as i32;
scatter_elem_sz_prev = (_AC!() << 12i32) as i32;
} else {
scatter_elem_sz_prev = *__slate_slot_16;
}
}
*__slate_slot_12 = get_order(*__slate_slot_16 as u64);
'__loop_14: loop {
ret_sz = 1i32 << 12i32.wrapping_add(*__slate_slot_12);
*__slate_slot_18 = 0;
*__slate_slot_17 = blk_size;
loop {
if *__slate_slot_17 > 0 && *__slate_slot_18 < *__slate_slot_15 {
*__slate_slot_16 = if *__slate_slot_17 > scatter_elem_sz_prev { scatter_elem_sz_prev } else { *__slate_slot_17 };
*unsafe { (*schp).pages }.wrapping_offset(*__slate_slot_18 as isize) = alloc_hooks__4!(alloc_pages_noprof(gfp_mask, *__slate_slot_12 as u32));
if unsafe { *unsafe { (*schp).pages }.wrapping_offset(*__slate_slot_18 as isize) }.is_null() {
break;
} else {
if *__slate_slot_16 == scatter_elem_sz_prev {
if unlikely!(ret_sz > scatter_elem_sz_prev) != 0 {
scatter_elem_sz = ret_sz;
scatter_elem_sz_prev = ret_sz;
}
}
SCSI_LOG_TIMEOUT!();
let __v7: i32 = *__slate_slot_18;
*__slate_slot_18 = __v7.wrapping_add(1);
let __v5: i32 = *__slate_slot_17;
*__slate_slot_17 = __v5.wrapping_sub(ret_sz);
}
} else {
break '__loop_14;
}
}
*__slate_slot_19 = 0;
while *__slate_slot_19 < *__slate_slot_18 {
__free_pages(*unsafe { (*schp).pages }.wrapping_offset(*__slate_slot_19 as isize), *__slate_slot_12 as u32);
let __v3: i32 = *__slate_slot_19;
*__slate_slot_19 = __v3.wrapping_add(1);
}
let __v1: i32 = *__slate_slot_12;
let __v0: i32 = __v1.wrapping_sub(1);
*__slate_slot_12 = __v0;
if __v0 < 0 {
return 12i32.wrapping_neg();
}
}
// end of for loop
(*schp).page_order = *__slate_slot_12;
(*schp).k_use_sg = *__slate_slot_18 as u16;
SCSI_LOG_TIMEOUT!();
(*schp).bufflen = blk_size as u32;
if *__slate_slot_17 > 0 {
return 12i32.wrapping_neg();
} else {
// must have failed
return 0;
}
}
}
}
}

fn sg_remove_scat(sfp: *mut sg_fd, schp: *mut sg_scatter_hold) {
SCSI_LOG_TIMEOUT!();
if !unsafe { (*schp).pages }.is_null() && (unsafe { (*schp).sglist_len }) > 0 {
if (unsafe { (*schp).dio_in_use }) == 0 {
let mut k: i32 = 0;
while k < ((unsafe { (*schp).k_use_sg }) as i32) && !unsafe { *unsafe { (*schp).pages }.wrapping_offset(k as isize) }.is_null() {
SCSI_LOG_TIMEOUT!();
unsafe { __free_pages(*unsafe { (*schp).pages }.wrapping_offset(k as isize), (*schp).page_order as u32) };
k = k.wrapping_add(1);
}
unsafe { kfree((*schp).pages as *const ()) };
}
}
unsafe { memset(schp as *mut (), 0i32, core::mem::size_of::<sg_scatter_hold>() as u64) };
}

fn sg_read_oxfer(srp: *mut sg_request, mut outp: *mut u8, mut num_read_xfer: i32) -> i32 {
let schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*srp).data) };
let mut k: i32 = 0;
let mut num: i32 = 0;
SCSI_LOG_TIMEOUT!();
if outp.is_null() || num_read_xfer <= 0 {
return 0;
}
num = 1i32 << 12i32.wrapping_add(unsafe { (*schp).page_order });
while k < ((unsafe { (*schp).k_use_sg }) as i32) && !unsafe { *unsafe { (*schp).pages }.wrapping_offset(k as isize) }.is_null() {
if num > num_read_xfer {
if copy_to_user(outp as *mut (), page_address__2!(unsafe { *unsafe { (*schp).pages }.wrapping_offset(k as isize) }) as *const (), num_read_xfer as u64) != 0 {
return 14i32.wrapping_neg();
}
break;
} else {
if copy_to_user(outp as *mut (), page_address__2!(unsafe { *unsafe { (*schp).pages }.wrapping_offset(k as isize) }) as *const (), num as u64) != 0 {
return 14i32.wrapping_neg();
}
num_read_xfer = num_read_xfer.wrapping_sub(num);
if num_read_xfer <= 0 {
break;
}
let __v4: *mut u8 = outp;
outp = __v4.wrapping_offset(num as isize);
}
k = k.wrapping_add(1);
}
0
}

fn sg_build_reserve(sfp: *mut sg_fd, mut req_size: i32) {
let schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*sfp).reserve) };
SCSI_LOG_TIMEOUT!();
loop {
if (req_size as u64) < _AC!() << 12i32 {
req_size = (_AC!() << 12i32) as i32;
}
if 0 == sg_build_indirect(schp, sfp, req_size) {
return;
} else {
sg_remove_scat(sfp, schp);
}
req_size >>= 1i32; // divide by 2
if (req_size as u64) <= (_AC!() << 12i32) / 2 {
break;
}
}
}

fn sg_link_reserve(sfp: *mut sg_fd, srp: *mut sg_request, size: i32) {
let req_schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*srp).data) };
let rsv_schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*sfp).reserve) };
let mut k: i32 = 0;
let mut num: i32 = 0;
let mut rem: i32 = 0;
unsafe {
(*srp).res_used = 1;
}
SCSI_LOG_TIMEOUT!();
rem = size;
num = 1i32 << 12i32.wrapping_add(unsafe { (*rsv_schp).page_order });
while k < ((unsafe { (*rsv_schp).k_use_sg }) as i32) {
if rem <= num {
unsafe {
(*req_schp).k_use_sg = k.wrapping_add(1) as u16;
}
unsafe {
(*req_schp).sglist_len = (*rsv_schp).sglist_len;
}
unsafe {
(*req_schp).pages = (*rsv_schp).pages;
}
unsafe {
(*req_schp).bufflen = size as u32;
}
unsafe {
(*req_schp).page_order = (*rsv_schp).page_order;
}
break;
} else {
rem = rem.wrapping_sub(num);
}
k = k.wrapping_add(1);
}
if k >= ((unsafe { (*rsv_schp).k_use_sg }) as i32) {
SCSI_LOG_TIMEOUT!();
}
}

fn sg_unlink_reserve(sfp: *mut sg_fd, srp: *mut sg_request) {
let req_schp: *mut sg_scatter_hold = unsafe { core::ptr::addr_of_mut!((*srp).data) };
SCSI_LOG_TIMEOUT!();
unsafe {
(*req_schp).k_use_sg = 0;
}
unsafe {
(*req_schp).bufflen = 0;
}
unsafe {
(*req_schp).pages = core::ptr::null_mut::<*mut page>();
}
unsafe {
(*req_schp).page_order = 0;
}
unsafe {
(*req_schp).sglist_len = 0;
}
unsafe {
(*srp).res_used = 0;
}
// Called without mutex lock to avoid deadlock
unsafe {
(*sfp).res_in_use = 0;
}
}

fn sg_get_rq_mark(sfp: *mut sg_fd, pack_id: i32, busy: &mut bool) -> *mut sg_request {
let mut resp: *mut sg_request = unsafe { core::mem::zeroed() };
let mut iflags: u64 = 0;
unsafe {
*(busy as *mut bool) = false;
}
write_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
resp = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) });
while list_entry_is_head!(resp, unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) }, entry) == 0 {
// look for requests that are not SG_IO owned
if (unsafe { (*resp).sg_io_owned }) == 0 && (1i32.wrapping_neg() == pack_id || (unsafe { (*resp).header.pack_id }) == pack_id) {
'__slate_break_3: {
match (unsafe { (*resp).done }) as i32 {
0 => {
unsafe {
*(busy as *mut bool) = true;
}
// request active
}
1 => {
unsafe {
(*resp).done = 2;
}
// request done; response ready to return
// guard against other readers
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
return resp;
}
2 => {
break '__slate_break_3; // response already being returned
}
_ => {
}
}
}
}
resp = list_next_entry__3!(resp, entry);
}
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
core::ptr::null_mut::<sg_request>()
}

/// always adds to end of list
fn sg_add_request(sfp: *mut sg_fd) -> *mut sg_request {
let mut rp: *mut sg_request = unsafe { core::mem::zeroed() };
let mut iflags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u64 = core::ptr::addr_of_mut!(iflags) as *mut u64;
let mut k: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(k) as *mut i32;
unsafe {
'__join_1: {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
rp = (*sfp).req_arr.as_mut_ptr() as *mut sg_request;
write_lock_irqsave!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_5);
if list_empty(core::ptr::addr_of_mut!((*sfp).rq_list) as *const list_head) == 0 {
if (*sfp).cmd_q != 0 {
*__slate_slot_6 = 0;
while *__slate_slot_6 < 16 {
if unsafe { (*rp).parentfp }.is_null() {
break;
} else {
let __v3: i32 = *__slate_slot_6;
*__slate_slot_6 = __v3.wrapping_add(1);
let __v1: *mut sg_request = rp;
rp = __v1.wrapping_offset(1 as isize);
}
}
if *__slate_slot_6 < 16 {
break '__join_1;
}
}
write_unlock_irqrestore!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_5);
return core::ptr::null_mut::<sg_request>();
}
}
memset(rp as *mut (), 0i32, core::mem::size_of::<sg_request>() as u64);
(*rp).parentfp = sfp;
(*rp).header.duration = jiffies_to_msecs(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)));
list_add_tail(core::ptr::addr_of_mut!((*rp).entry), core::ptr::addr_of_mut!((*sfp).rq_list));
write_unlock_irqrestore!(core::ptr::addr_of_mut!((*sfp).rq_list_lock), *__slate_slot_5);
return rp;
}
}

/// Return of 1 for found; 0 for not found
fn sg_remove_request(sfp: *mut sg_fd, srp: *mut sg_request) -> i32 {
let mut iflags: u64 = 0;
let mut res: i32 = 0;
let __v0: bool = sfp.is_null() || srp.is_null() || list_empty((unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) }) as *const list_head) != 0;
if __v0 {
return res;
}
write_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
if list_empty((unsafe { core::ptr::addr_of_mut!((*srp).entry) }) as *const list_head) == 0 {
list_del(unsafe { core::ptr::addr_of_mut!((*srp).entry) });
unsafe {
(*srp).parentfp = core::ptr::null_mut::<sg_fd>();
}
res = 1;
}
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
// If the device is detaching, wakeup any readers in case we just
// removed the last response, which would leave nothing for them to
// return other than -ENODEV.
if unlikely__2!(atomic_read((unsafe { core::ptr::addr_of_mut!((*(*sfp).parentdp).detaching) }) as *const __SlateRecord_types_h_190)) != 0 {
wake_up_interruptible_all!(unsafe { core::ptr::addr_of_mut!((*sfp).read_wait) });
}
res
}

fn sg_add_sfp(sdp: *mut sg_device) -> *mut sg_fd {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: rwlock = unsafe { core::mem::zeroed() };
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let mut iflags: u64 = 0;
let mut bufflen: i32 = 0;
let mut __v0: *mut sg_fd;
let __obj_size: u64 = size_mul(core::mem::size_of::<sg_fd>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, 10272u32, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, 10272u32, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut sg_fd;
sfp = __v0;
if sfp.is_null() {
return ERR_PTR(12i32.wrapping_neg() as i64) as *mut sg_fd;
}
loop {
unsafe { __init_waitqueue_head(core::ptr::addr_of_mut!((*sfp).read_wait), c"&sfp->read_wait".as_ptr() as *const u8, core::ptr::addr_of_mut!(__key__2153)) };
if true {
break;
}
}
'__slate_break_8: loop {
unsafe {
*core::ptr::addr_of_mut!((*sfp).rq_list_lock) = *{
__slate_compound_7 = rwlock { raw_lock: qrwlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qrwlock_types_h_27 = core::mem::zeroed();
__t0.cnts = __SlateRecord_types_h_190 { counter: 0 };
__t0
}, wait_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t1.val = __SlateRecord_types_h_190 { counter: 0 };
__t1
} } } };
core::ptr::addr_of_mut!(__slate_compound_7)
};
}
if true {
break;
}
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) });
kref_init(unsafe { core::ptr::addr_of_mut!((*sfp).f_ref) });
loop {
__mutex_init(unsafe { core::ptr::addr_of_mut!((*sfp).f_mutex) }, c"&sfp->f_mutex".as_ptr() as *const u8, unsafe { core::ptr::addr_of_mut!(__key__2157) });
if true {
break;
}
}
unsafe {
(*sfp).timeout = mult_frac!(6000i32, 1000i32, 100i32);
}
unsafe {
(*sfp).timeout_user = 6000i32;
}
unsafe {
(*sfp).force_packid = 0;
}
unsafe {
(*sfp).cmd_q = 0;
}
unsafe {
(*sfp).keep_orphan = 0;
}
unsafe {
(*sfp).parentdp = sdp;
}
write_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sdp).sfd_lock) }, iflags);
if atomic_read((unsafe { core::ptr::addr_of_mut!((*sdp).detaching) }) as *const __SlateRecord_types_h_190) != 0 {
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sdp).sfd_lock) }, iflags);
unsafe { kfree(sfp as *const ()) };
return ERR_PTR(19i32.wrapping_neg() as i64) as *mut sg_fd;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*sfp).sfd_siblings) }, unsafe { core::ptr::addr_of_mut!((*sdp).sfds) });
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sdp).sfd_lock) }, iflags);
SCSI_LOG_TIMEOUT!();
bufflen = min_t!(i32, unsafe { def_reserved_size }, max_sectors_bytes(unsafe { (*(*sdp).device).request_queue }));
sg_build_reserve(sfp, bufflen);
SCSI_LOG_TIMEOUT!();
kref_get(unsafe { core::ptr::addr_of_mut!((*sdp).d_ref) });
unsafe { __module_get(core::ptr::null_mut::<module>()) };
sfp
}

static mut __key__2153: lock_class_key = unsafe { core::mem::zeroed() };

static mut __key__2157: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.sg.sg_remove_sfp_usercontext")]
extern "C" fn sg_remove_sfp_usercontext(work: *mut work_struct) {
let mut sfp: *mut sg_fd = unsafe { core::mem::zeroed() };
let mut __v0: *mut sg_fd;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__8!(sg_fd) as isize)) }) as *mut ()) as *mut sg_fd;
sfp = __v0;
let sdp: *mut sg_device = unsafe { (*sfp).parentdp };
let device: *mut scsi_device = unsafe { (*sdp).device };
let mut srp: *mut sg_request = unsafe { core::mem::zeroed() };
let mut iflags: u64 = 0;
// Cleanup any responses which were never read().
write_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
while list_empty((unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) }) as *const list_head) == 0 {
srp = list_first_entry__10!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list) }, sg_request);
list_del(unsafe { core::ptr::addr_of_mut!((*srp).entry) });
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
sg_finish_rem_req(srp);
// sg_rq_end_io() uses srp->parentfp. Hence, only clear
// srp->parentfp after blk_mq_free_request() has been called.
unsafe {
(*srp).parentfp = core::ptr::null_mut::<sg_fd>();
}
write_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
}
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sfp).rq_list_lock) }, iflags);
if (unsafe { (*sfp).reserve.bufflen }) > 0 {
SCSI_LOG_TIMEOUT!();
sg_remove_scat(sfp, unsafe { core::ptr::addr_of_mut!((*sfp).reserve) });
}
SCSI_LOG_TIMEOUT!();
unsafe { kfree(sfp as *const ()) };
kref_put(unsafe { core::ptr::addr_of_mut!((*sdp).d_ref) }, Some(sg_device_destroy));
unsafe { scsi_device_put(device) };
unsafe { module_put(core::ptr::null_mut::<module>()) };
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_remove_sfp")]
extern "C" fn sg_remove_sfp(kref: *mut kref) {
let mut __slate_compound_5: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let sfp: *mut sg_fd = container_of__25!(kref, sg_fd);
let sdp: *mut sg_device = unsafe { (*sfp).parentdp };
let mut iflags: u64 = 0;
write_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sdp).sfd_lock) }, iflags);
list_del(unsafe { core::ptr::addr_of_mut!((*sfp).sfd_siblings) });
write_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!((*sdp).sfd_lock) }, iflags);
'__slate_break_7: loop {
'__slate_break_6: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*sfp).ew.work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*sfp).ew.work)).data = *{
__slate_compound_5 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_5)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*sfp).ew.work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*sfp).ew.work)).func = Some(sg_remove_sfp_usercontext);
}
if true {
break;
}
}
if true {
break;
}
}
schedule_work(unsafe { core::ptr::addr_of_mut!((*sfp).ew.work) });
}

static mut __key__2243: lock_class_key = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.sg.sg_idr_max_id")]
extern "C" fn sg_idr_max_id(id: i32, p: *mut (), data: *mut ()) -> i32 {
if (unsafe { *(data as *mut i32) }) < id {
unsafe {
*(data as *mut i32) = id;
}
}
0
}

fn sg_last_dev() -> i32 {
let mut k: i32 = 1i32.wrapping_neg();
let mut iflags: u64 = 0;
read_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, iflags);
unsafe { idr_for_each(core::ptr::addr_of_mut!(sg_index_idr) as *const idr, Some(sg_idr_max_id), core::ptr::addr_of_mut!(k) as *mut ()) };
read_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, iflags);
return k.wrapping_add(1); // origin 1
}

/// must be called with sg_index_lock held
fn sg_lookup_dev(dev: i32) -> *mut sg_device {
(unsafe { idr_find(core::ptr::addr_of_mut!(sg_index_idr) as *const idr, dev as u64) }) as *mut sg_device
}

fn sg_get_dev(dev: i32) -> *mut sg_device {
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
read_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, flags);
sdp = sg_lookup_dev(dev);
if sdp.is_null() {
sdp = ERR_PTR(6i32.wrapping_neg() as i64) as *mut sg_device;
} else {
if atomic_read((unsafe { core::ptr::addr_of_mut!((*sdp).detaching) }) as *const __SlateRecord_types_h_190) != 0 {
// If sdp->detaching, then the refcount may already be 0, in
// which case it would be a bug to do kref_get().
sdp = ERR_PTR(19i32.wrapping_neg() as i64) as *mut sg_device;
} else {
kref_get(unsafe { core::ptr::addr_of_mut!((*sdp).d_ref) });
}
}
read_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, flags);
sdp
}

static adio_proc_ops: __SlateConst<proc_ops> = __SlateConst({
let mut __t0: proc_ops = unsafe { core::mem::zeroed() };
__t0.proc_open = Some(sg_proc_single_open_adio);
__t0.proc_read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.proc_write = Some(sg_proc_write_adio);
__t0.proc_lseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.proc_release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

static dressz_proc_ops: __SlateConst<proc_ops> = __SlateConst({
let mut __t0: proc_ops = unsafe { core::mem::zeroed() };
__t0.proc_open = Some(sg_proc_single_open_dressz);
__t0.proc_read = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, *mut u8, u64, *mut i64) -> i64>>(seq_read as *const ()) };
__t0.proc_write = Some(sg_proc_write_dressz);
__t0.proc_lseek = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut file, i64, i32) -> i64>>(seq_lseek as *const ()) };
__t0.proc_release = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut inode, *mut file) -> i32>>(single_release as *const ()) };
__t0
});

static dev_seq_ops: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(dev_seq_start), stop: Some(dev_seq_stop), next: Some(dev_seq_next), show: Some(sg_proc_seq_show_dev) });

static devstrs_seq_ops: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(dev_seq_start), stop: Some(dev_seq_stop), next: Some(dev_seq_next), show: Some(sg_proc_seq_show_devstrs) });

static debug_seq_ops: __SlateConst<seq_operations> = __SlateConst(seq_operations { start: Some(dev_seq_start), stop: Some(dev_seq_stop), next: Some(dev_seq_next), show: Some(sg_proc_seq_show_debug) });

fn sg_proc_init() -> i32 {
let p: *mut proc_dir_entry = unsafe { proc_mkdir(c"scsi/sg".as_ptr() as *const u8, core::ptr::null_mut::<proc_dir_entry>()) };
if p.is_null() {
return 1;
}
unsafe { proc_create(c"allow_dio".as_ptr() as *const u8, 420u16, p, core::ptr::addr_of!(adio_proc_ops.0)) };
proc_create_seq!(c"debug".as_ptr() as *mut u8, 292i32, p, unsafe { core::ptr::addr_of!(debug_seq_ops.0) });
unsafe { proc_create(c"def_reserved_size".as_ptr() as *const u8, 420u16, p, core::ptr::addr_of!(dressz_proc_ops.0)) };
proc_create_single!(c"device_hdr".as_ptr() as *mut u8, 292i32, p, Some(sg_proc_seq_show_devhdr));
proc_create_seq!(c"devices".as_ptr() as *mut u8, 292i32, p, unsafe { core::ptr::addr_of!(dev_seq_ops.0) });
proc_create_seq!(c"device_strs".as_ptr() as *mut u8, 292i32, p, unsafe { core::ptr::addr_of!(devstrs_seq_ops.0) });
proc_create_single!(c"version".as_ptr() as *mut u8, 292i32, p, Some(sg_proc_seq_show_version));
0
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_seq_show_int")]
extern "C" fn sg_proc_seq_show_int(s: *mut seq_file, v: *mut ()) -> i32 {
unsafe { seq_printf(s, c"%d\n".as_ptr() as *const u8, *((*s).private as *mut i32)) };
0
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_single_open_adio")]
extern "C" fn sg_proc_single_open_adio(inode: *mut inode, file: *mut file) -> i32 {
unsafe { single_open(file, Some(sg_proc_seq_show_int), core::ptr::addr_of_mut!(sg_allow_dio) as *mut ()) }
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_write_adio")]
extern "C" fn sg_proc_write_adio(filp: *mut file, buffer: *const u8, count: u64, off: *mut i64) -> i64 {
let mut err: i32 = 0;
let mut num: u64 = 0;
let __v0: bool = !unsafe { capable(21i32) } || !unsafe { capable(17i32) };
if __v0 {
return 13i32.wrapping_neg() as i64;
}
err = unsafe { kstrtoul_from_user(buffer, count, 0u32, core::ptr::addr_of_mut!(num)) };
if err != 0 {
return err as i64;
}
unsafe {
sg_allow_dio = if num != 0 { 1 } else { 0 };
}
count as i64
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_single_open_dressz")]
extern "C" fn sg_proc_single_open_dressz(inode: *mut inode, file: *mut file) -> i32 {
unsafe { single_open(file, Some(sg_proc_seq_show_int), core::ptr::addr_of_mut!(def_reserved_size) as *mut ()) }
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_write_dressz")]
extern "C" fn sg_proc_write_dressz(filp: *mut file, buffer: *const u8, count: u64, off: *mut i64) -> i64 {
let mut err: i32 = 0;
let mut k: u64 = !0;
let __v0: bool = !unsafe { capable(21i32) } || !unsafe { capable(17i32) };
if __v0 {
return 13i32.wrapping_neg() as i64;
}
err = unsafe { kstrtoul_from_user(buffer, count, 0u32, core::ptr::addr_of_mut!(k)) };
if err != 0 {
return err as i64;
}
if k <= 1048576 {
// limit "big buff" to 1 MB
unsafe {
def_reserved_size = k as i32;
}
return count as i64;
}
34i32.wrapping_neg() as i64
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_seq_show_version")]
extern "C" fn sg_proc_seq_show_version(s: *mut seq_file, v: *mut ()) -> i32 {
unsafe { seq_printf(s, c"%d\t%s [%s]\n".as_ptr() as *const u8, sg_version_num, c"3.5.36".as_ptr() as *mut u8, sg_version_date) };
0
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_seq_show_devhdr")]
extern "C" fn sg_proc_seq_show_devhdr(s: *mut seq_file, v: *mut ()) -> i32 {
seq_puts(s, c"host\tchan\tid\tlun\ttype\topens\tqdepth\tbusy\tonline\n".as_ptr() as *const u8);
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sg_proc_deviter {
    index: i64,
    max: u64,
}


#[unsafe(link_section = ".text.slate_distinct.sg.dev_seq_start")]
extern "C" fn dev_seq_start(s: *mut seq_file, pos: *mut i64) -> *mut () {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: *mut sg_proc_deviter;
let __obj_size: u64 = size_mul(core::mem::size_of::<sg_proc_deviter>() as u64, 1u64);
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
__v0 = __v1 as *mut sg_proc_deviter;
let it: *mut sg_proc_deviter = __v0;
unsafe {
(*s).private = it as *mut ();
}
if it.is_null() {
return core::ptr::null_mut::<()>();
}
unsafe {
(*it).index = *pos;
}
unsafe {
(*it).max = sg_last_dev() as u64;
}
if ((unsafe { (*it).index }) as u64) >= unsafe { (*it).max } {
return core::ptr::null_mut::<()>();
}
it as *mut ()
}

#[unsafe(link_section = ".text.slate_distinct.sg.dev_seq_next")]
extern "C" fn dev_seq_next(s: *mut seq_file, v: *mut (), pos: *mut i64) -> *mut () {
let it: *mut sg_proc_deviter = (unsafe { (*s).private }) as *mut sg_proc_deviter;
let __v1: i64 = unsafe { (*it).index };
let __v2: i64 = __v1.wrapping_add(1);
unsafe {
(*it).index = __v2;
}
unsafe {
*pos = __v2;
}
(if ((unsafe { (*it).index }) as u64) < unsafe { (*it).max } { it } else { core::ptr::null_mut::<sg_proc_deviter>() }) as *mut ()
}

#[unsafe(link_section = ".text.slate_distinct.sg.dev_seq_stop")]
extern "C" fn dev_seq_stop(s: *mut seq_file, v: *mut ()) {
unsafe { kfree((*s).private as *const ()) };
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_seq_show_dev")]
extern "C" fn sg_proc_seq_show_dev(s: *mut seq_file, v: *mut ()) -> i32 {
let it: *mut sg_proc_deviter = v as *mut sg_proc_deviter;
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut iflags: u64 = 0;
read_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, iflags);
let __v2: *mut sg_device = if !it.is_null() {
sg_lookup_dev((unsafe { (*it).index }) as i32)
} else {
core::ptr::null_mut::<sg_device>()
};
sdp = __v2;
let __v3: bool = sdp.is_null() || ((unsafe { (*sdp).device }) as *mut ()).is_null() || atomic_read((unsafe { core::ptr::addr_of_mut!((*sdp).detaching) }) as *const __SlateRecord_types_h_190) != 0;
if __v3 {
seq_puts(s, c"-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\t-1\n".as_ptr() as *const u8);
} else {
let scsidp: *mut scsi_device = unsafe { (*sdp).device };
unsafe { seq_printf(s, c"%d\t%d\t%d\t%llu\t%d\t%d\t%d\t%d\t%d\n".as_ptr() as *const u8, (*(*scsidp).host).host_no, (*scsidp).channel, (*scsidp).id, (*scsidp).lun, (*scsidp).r#type as i32, 1i32, (*scsidp).queue_depth as i32, scsi_device_busy(scsidp), scsi_device_online(scsidp)) };
}
read_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, iflags);
0
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_seq_show_devstrs")]
extern "C" fn sg_proc_seq_show_devstrs(s: *mut seq_file, v: *mut ()) -> i32 {
let it: *mut sg_proc_deviter = v as *mut sg_proc_deviter;
let mut sdp: *mut sg_device = unsafe { core::mem::zeroed() };
let mut iflags: u64 = 0;
read_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, iflags);
let __v2: *mut sg_device = if !it.is_null() {
sg_lookup_dev((unsafe { (*it).index }) as i32)
} else {
core::ptr::null_mut::<sg_device>()
};
sdp = __v2;
let scsidp: *mut scsi_device = if !sdp.is_null() { unsafe { (*sdp).device } } else { core::ptr::null_mut::<scsi_device>() };
let __v3: bool = !sdp.is_null() && !scsidp.is_null() && atomic_read((unsafe { core::ptr::addr_of_mut!((*sdp).detaching) }) as *const __SlateRecord_types_h_190) == 0;
if __v3 {
unsafe { seq_printf(s, c"%8.8s\t%16.16s\t%4.4s\n".as_ptr() as *const u8, (*scsidp).vendor.as_mut_ptr() as *mut u8, (*scsidp).model.as_mut_ptr() as *mut u8, (*scsidp).rev.as_mut_ptr() as *mut u8) };
} else {
seq_puts(s, c"<no active device>\n".as_ptr() as *const u8);
}
read_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!(sg_index_lock) }, iflags);
0
}

// must be called while holding sg_index_lock
fn sg_proc_debug_helper(s: *mut seq_file, sdp: *mut sg_device) {
let mut k: i32 = 0;
let mut new_interface: i32 = 0;
let mut blen: i32 = 0;
let mut usg: i32 = 0;
let mut srp: *mut sg_request = unsafe { core::mem::zeroed() };
let mut fp: *mut sg_fd = unsafe { core::mem::zeroed() };
let mut hp: *const sg_io_hdr = unsafe { core::mem::zeroed() };
let mut cp: *const u8 = unsafe { core::mem::zeroed() };
let mut ms: u32 = 0;
let mut duration: u32 = 0;
fp = list_first_entry__5!(unsafe { core::ptr::addr_of_mut!((*sdp).sfds) });
while list_entry_is_head!(fp, unsafe { core::ptr::addr_of_mut!((*sdp).sfds) }, sfd_siblings) == 0 {
k = k.wrapping_add(1);
read_lock!(unsafe { core::ptr::addr_of_mut!((*fp).rq_list_lock) }); // irqs already disabled
unsafe { seq_printf(s, c"   FD(%d): timeout=%dms bufflen=%d (res)sgat=%d low_dma=%d\n".as_ptr() as *const u8, k, jiffies_to_msecs((*fp).timeout as u64), (*fp).reserve.bufflen, (*fp).reserve.k_use_sg as i32, 0i32) };
unsafe { seq_printf(s, c"   cmd_q=%d f_packid=%d k_orphan=%d closed=0\n".as_ptr() as *const u8, (*fp).cmd_q as i32, (*fp).force_packid as i32, (*fp).keep_orphan as i32) };
srp = list_first_entry__3!(unsafe { core::ptr::addr_of_mut!((*fp).rq_list) });
while list_entry_is_head!(srp, unsafe { core::ptr::addr_of_mut!((*fp).rq_list) }, entry) == 0 {
hp = (unsafe { core::ptr::addr_of_mut!((*srp).header) }) as *const sg_io_hdr;
new_interface = if (unsafe { (*hp).interface_id }) == 0 { 0 } else { 1 };
if (unsafe { (*srp).res_used }) != 0 {
if new_interface != 0 && 4 & unsafe { (*hp).flags } != 0 {
cp = c"     mmap>> ".as_ptr() as *const u8;
} else {
cp = c"     rb>> ".as_ptr() as *const u8;
}
} else {
if 6 & unsafe { (*hp).info } != 0 {
cp = c"     dio>> ".as_ptr() as *const u8;
} else {
cp = c"     ".as_ptr() as *const u8;
}
}
seq_puts(s, cp);
blen = (unsafe { (*srp).data.bufflen }) as i32;
usg = (unsafe { (*srp).data.k_use_sg }) as i32;
seq_puts(s, (if (unsafe { (*srp).done }) != 0 { if 1 == ((unsafe { (*srp).done }) as i32) { c"rcv:".as_ptr() as *mut u8 } else { c"fin:".as_ptr() as *mut u8 } } else { c"act:".as_ptr() as *mut u8 }) as *const u8);
unsafe { seq_printf(s, c" id=%d blen=%d".as_ptr() as *const u8, (*srp).header.pack_id, blen) };
if (unsafe { (*srp).done }) != 0 {
unsafe { seq_printf(s, c" dur=%u".as_ptr() as *const u8, (*hp).duration) };
} else {
ms = jiffies_to_msecs(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) });
let mut __v4: u32;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_622() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*hp).duration) as *const u32) };
duration = __v4;
if duration != 0 {
duration = if ms > duration { ms.wrapping_sub(duration) } else { 0 };
}
let __v6: u32 = if new_interface != 0 {
unsafe { (*hp).timeout }
} else {
jiffies_to_msecs((unsafe { (*fp).timeout }) as u64)
};
unsafe { seq_printf(s, c" t_o/elap=%u/%u".as_ptr() as *const u8, __v6, duration) };
}
unsafe { seq_printf(s, c"ms sgat=%d op=0x%02x\n".as_ptr() as *const u8, usg, (*srp).data.cmd_opcode as i32) };
srp = list_next_entry__3!(srp, entry);
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*fp).rq_list) }) as *const list_head) != 0 {
seq_puts(s, c"     No requests active\n".as_ptr() as *const u8);
}
read_unlock!(unsafe { core::ptr::addr_of_mut!((*fp).rq_list_lock) });
fp = list_next_entry__5!(fp, sfd_siblings);
}
}

#[unsafe(link_section = ".text.slate_distinct.sg.sg_proc_seq_show_debug")]
extern "C" fn sg_proc_seq_show_debug(s: *mut seq_file, v: *mut ()) -> i32 {
let mut __v1: *mut sg_device = unsafe { core::mem::zeroed() };
let mut iflags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u64 = core::ptr::addr_of_mut!(iflags) as *mut u64;
let mut sdp: core::mem::MaybeUninit<*mut sg_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut sg_device = core::ptr::addr_of_mut!(sdp) as *mut *mut sg_device;
let mut it: *mut sg_proc_deviter = unsafe { core::mem::zeroed() };
unsafe {
it = v as *mut sg_proc_deviter;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if !it.is_null() && 0 == (*it).index {
seq_printf(s, c"max_active_device=%d  def_reserved_size=%d\n".as_ptr() as *const u8, (*it).max as i32, def_reserved_size);
}
read_lock_irqsave!(core::ptr::addr_of_mut!(sg_index_lock), *__slate_slot_2);
if !it.is_null() {
__v1 = sg_lookup_dev((*it).index as i32);
} else {
__v1 = core::ptr::null_mut::<sg_device>();
}
*__slate_slot_3 = __v1;
if !(*__slate_slot_3 as *mut ()).is_null() {
read_lock!(core::ptr::addr_of_mut!((*(*__slate_slot_3)).sfd_lock));
if list_empty(core::ptr::addr_of_mut!((*(*__slate_slot_3)).sfds) as *const list_head) == 0 {
seq_printf(s, c" >>> device=%s ".as_ptr() as *const u8, (*(*__slate_slot_3)).name.as_mut_ptr() as *mut u8);
if atomic_read(core::ptr::addr_of_mut!((*(*__slate_slot_3)).detaching) as *const __SlateRecord_types_h_190) != 0 {
seq_puts(s, c"detaching pending close ".as_ptr() as *const u8);
} else {
if !unsafe { (*(*__slate_slot_3)).device }.is_null() {
let scsidp: *mut scsi_device = (*(*__slate_slot_3)).device;
seq_printf(s, c"%d:%d:%d:%llu   em=%d".as_ptr() as *const u8, (*(*scsidp).host).host_no, (*scsidp).channel, (*scsidp).id, (*scsidp).lun, (*(*(*scsidp).host).hostt).__slate_bits_0.__get_emulated() as i32);
}
}
seq_printf(s, c" sg_tablesize=%d excl=%d open_cnt=%d\n".as_ptr() as *const u8, (*(*__slate_slot_3)).sg_tablesize, (*(*__slate_slot_3)).exclude as i32, (*(*__slate_slot_3)).open_cnt);
sg_proc_debug_helper(s, *__slate_slot_3);
}
read_unlock!(core::ptr::addr_of_mut!((*(*__slate_slot_3)).sfd_lock));
}
read_unlock_irqrestore!(core::ptr::addr_of_mut!(sg_index_lock), *__slate_slot_2);
return 0;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_sg_624: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(init_sg)) };

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_exit_sg: Option<unsafe extern "C" fn()> = Some(exit_sg);

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_get_page_1: [u8; 1] = *b"\0";

static __slate_asm_string_get_page_2: [u8; 48] = *b"/home/takashi/c-corpus/linux/include/linux/mm.h\0";

static __slate_asm_string_get_page_6: [u8; 1] = *b"\0";

static __slate_asm_string_get_page_7: [u8; 48] = *b"/home/takashi/c-corpus/linux/include/linux/mm.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_rwsem_assert_held_write_nolockdep_1: [u8; 1] = *b"\0";

static __slate_asm_string_rwsem_assert_held_write_nolockdep_2: [u8; 51] = *b"/home/takashi/c-corpus/linux/include/linux/rwsem.h\0";

static __slate_asm_string_sg_rq_end_io_1: [u8; 1] = *b"\0";

static __slate_asm_string_sg_rq_end_io_2: [u8; 47] = *b"/home/takashi/c-corpus/linux/drivers/scsi/sg.c\0";

static __slate_asm_string_sg_rq_end_io_6: [u8; 1] = *b"\0";

static __slate_asm_string_sg_rq_end_io_7: [u8; 47] = *b"/home/takashi/c-corpus/linux/drivers/scsi/sg.c\0";

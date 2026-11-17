
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
//! SPDX-License-Identifier: GPL-2.0-only
//!
//!     sd.c Copyright (C) 1992 Drew Eckhardt
//!          Copyright (C) 1993, 1994, 1995, 1999 Eric Youngdale
//!
//!     Linux scsi disk driver
//!             Initial versions: Drew Eckhardt
//!             Subsequent revisions: Eric Youngdale
//! Modification history:
//!      - Drew Eckhardt <drew@colorado.edu> original
//!      - Eric Youngdale <eric@andante.org> add scatter-gather, multiple
//!        outstanding request, and other enhancements.
//!        Support loadable low-level scsi drivers.
//!      - Jirka Hanika <geo@ff.cuni.cz> support more scsi disks using
//!        eight major numbers.
//!      - Richard Gooch <rgooch@atnf.csiro.au> support devfs.
//!  - Torben Mathiasen <tmm@image.dk> Resource allocation fixes in
//!    sd_init and cleanups.
//!  - Alex Davis <letmein@erols.com> Fix problem where partition info
//!    not being read in sd_open. Fix problem where removable media
//!    could be ejected after sd_open.
//!  - Douglas Gilbert <dgilbert@interlog.com> cleanup for lk 2.5.x
//!  - Badari Pulavarty <pbadari@us.ibm.com>, Matthew Wilcox
//!    <willy@debian.org>, Kurt Garloff <garloff@suse.de>:
//!    Support 32k/1M disks.
//!
//! Logging policy (needs CONFIG_SCSI_LOGGING defined):
//!  - setting up transfer: SCSI_LOG_HLQUEUE levels 1 and 2
//!  - end of transfer (bh + scsi_lib): SCSI_LOG_HLCOMPLETE level 1
//!  - entering sd_ioctl: SCSI_LOG_IOCTL level 1
//!  - entering other commands: SCSI_LOG_HLQUEUE level 3
//! Note: when the logging level is set by the user, it must be greater
//! than the level indicated above to trigger output.
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
//! SCHED_ISO: reserved but not implemented yet
//! Optimization barrier
//!
//! The "volatile" is due to gcc bugs
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
//! The generic version tends to create spurious ENDBR instructions under
//! certain conditions.
//! This counts to 15. Any more, it will return 16th argument.
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
//! These are for everybody (although not all archs will actually
//! discard it in modules)
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
//! error conditions
//! Internal kernel timer frequency
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
//! writeable
//! Operation not permitted
//! Value too large for defined data type
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
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
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
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
//! These are non-NULL pointers that will result in page faults
//! under normal circumstances, used to verify that nobody uses
//! non-initialized list entries.
//! list_entry - get the struct for this entry
//! @ptr:	the &struct list_head pointer.
//! @type:	the type of the struct this is embedded in.
//! @member:	the name of the list_head within the struct.
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
//! We use the MSB for PREEMPT_NEED_RESCHED mostly because it is available.
//! Macros to retrieve the current execution context:
//!
//! in_nmi()		- We're in NMI context
//! in_hardirq()		- We're in hard IRQ context
//! in_serving_softirq()	- We're in softirq context
//! in_task()		- We're in task context
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
//! wait_queue_entry::flags
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
//! Must be a compile-time constant, so implement it as a macro
//! Resource type
//! Bits set in the VMA until the stack is in its final location
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
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
//! iterate over multi-page bvec
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! Sizes of name fields. Unused at present.
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! Runtime PM flag argument bits
//! See the comment in asm-generic/io.h about ioremap_np().
//! values for service action in
//! The Well Known LUNS (SAM-3) in our int representation of a LUN
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! Do not query the IO Advice Hints Grouping mode page
//! More than enough for everybody ;)  The huge number of majors
//! is a leftover from 16bit dev_t days, we don't really need that
//! much numberspace.
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
// disable_context_analysis() - disables context analysis
//
// Disables context analysis. Must be paired with a later
// enable_context_analysis().
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
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
// warning conditions
// normal but significant condition
// informational
// debug-level messages
// Annotation for a "continued" line of log printout (only done after a
// line that had no enclosing \n). Only to be used by core/arch code
// during early bootup (a continued line is not SMP-safe otherwise).
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
// Read-only file system
// Operation not supported on transport endpoint
// No medium found
// Memory page has hardware error
// restart by calling sys_restart_syscall
// Operation is not supported
// IS_ERR_VALUE - Detect an error pointer.
// @x: The pointer to check.
//
// Like IS_ERR(), but does not generate a compiler warning if result is unused.
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
// min3 - return minimum of three values
// @x: first value
// @y: second value
// @z: third value
// max_t - return maximum of two values, using the specified type
// @type: data type to use
// @x: first value
// @y: second value
// min_not_zero - return the minimum that is _not_ zero, unless both are zero
// @x: value1
// @y: value2
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// Generate SD flag bits
// Implementation specific max size
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// ignore existing key
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
// Optional: only supported since gcc >= 15, clang >= 19
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Other-Builtins.html#index-_005f_005fbuiltin_005fcounted_005fby_005fref
// clang: https://clang.llvm.org/docs/LanguageExtensions.html#builtin-counted-by-ref
macro_rules! ALIGN_DOWN {
($x:expr, $a:expr) => {
__ALIGN_KERNEL!((($x) as u64).wrapping_sub($a.wrapping_sub(1)), $a)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__2 {
() => {
((core::mem::size_of::<[*const u8; 4]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__26!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__5 {
() => {
((core::mem::size_of::<[*const u8; 4]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__18!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__12 {
() => {
((core::mem::size_of::<[u8; 4]>() as u64) / (core::mem::size_of::<u8>() as u64)).wrapping_add(__must_be_array__29!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__26 {
() => {
((core::mem::size_of::<[*const u8; 6]>() as u64) / (core::mem::size_of::<*const u8>() as u64)).wrapping_add(__must_be_array__30!() as u64)
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

macro_rules! IOPRIO_PRIO_HINT {
($ioprio:expr) => {
(($ioprio) as i32) >> 3i32 & 1023i32
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

macro_rules! PFN_PHYS {
($x:expr) => {
(($x) as u64) << 12i32
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

macro_rules! SCSI_CHECK_LOGGING {
() => {
while false {
}
};
}

macro_rules! SCSI_LOG_HLCOMPLETE {
() => {
SCSI_CHECK_LOGGING!();
};
}

macro_rules! SCSI_LOG_HLQUEUE {
() => {
SCSI_CHECK_LOGGING!();
};
}

macro_rules! SCSI_LOG_IOCTL {
() => {
SCSI_CHECK_LOGGING!();
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

macro_rules! __ALIGN_KERNEL {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u64).wrapping_sub(1))
};
}

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
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
($x:expr) => {
__acquire_shared_ctx_lock__6($x);
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

/// __get_unaligned_t - read an unaligned value from memory.
/// @type:	the type to load from the pointer.
/// @ptr:	the pointer to load from.
///
/// Use memcpy to affect an unaligned type sized load avoiding undefined behavior
/// from approaches like type punning that require -fno-strict-aliasing in order
/// to be correct. As type may be const, use __unqual_scalar_typeof to map to a
/// non-const type - you can't memcpy into a const type. The
/// __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
/// expression rather than type, a pointer is used to avoid warnings about mixing
/// the use of 0 and NULL. The void* cast silences ubsan warnings.
macro_rules! __get_unaligned_t {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u32>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<$type>() as u64) };
__v = __v3;
__v
}
};
}

/// __get_unaligned_t - read an unaligned value from memory.
/// @type:	the type to load from the pointer.
/// @ptr:	the pointer to load from.
///
/// Use memcpy to affect an unaligned type sized load avoiding undefined behavior
/// from approaches like type punning that require -fno-strict-aliasing in order
/// to be correct. As type may be const, use __unqual_scalar_typeof to map to a
/// non-const type - you can't memcpy into a const type. The
/// __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
/// expression rather than type, a pointer is used to avoid warnings about mixing
/// the use of 0 and NULL. The void* cast silences ubsan warnings.
macro_rules! __get_unaligned_t__3 {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u64>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<$type>() as u64) };
__v = __v3;
__v
}
};
}

/// __get_unaligned_t - read an unaligned value from memory.
/// @type:	the type to load from the pointer.
/// @ptr:	the pointer to load from.
///
/// Use memcpy to affect an unaligned type sized load avoiding undefined behavior
/// from approaches like type punning that require -fno-strict-aliasing in order
/// to be correct. As type may be const, use __unqual_scalar_typeof to map to a
/// non-const type - you can't memcpy into a const type. The
/// __get_unaligned_ctrl_type gives __unqual_scalar_typeof its required
/// expression rather than type, a pointer is used to avoid warnings about mixing
/// the use of 0 and NULL. The void* cast silences ubsan warnings.
macro_rules! __get_unaligned_t__2 {
($type:ty, $ptr:expr) => {
{
let mut __v: $type;
core::ptr::null_mut::<u16>();
let mut __v3: $type = 0;
unsafe { __builtin_memcpy(core::ptr::addr_of_mut!(__v3) as *mut (), (($ptr) as *mut ()) as *const (), core::mem::size_of::<$type>() as u64) };
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

macro_rules! __is_nonneg {
($ux:expr) => {
statically_true!((($ux) as i64) >= 0)
};
}

macro_rules! __must_be_array__18 {
() => {
core::mem::size_of::<__SlateRecord_sd_c_246>() as i32
};
}

macro_rules! __must_be_array__26 {
() => {
core::mem::size_of::<__SlateRecord_sd_c_629>() as i32
};
}

macro_rules! __must_be_array__29 {
() => {
core::mem::size_of::<__SlateRecord_scsi_host_h_909>() as i32
};
}

macro_rules! __must_be_array__30 {
() => {
core::mem::size_of::<__SlateRecord_sd_c_589>() as i32
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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

/// __put_unaligned_t - write an unaligned value to memory.
/// @type:	the type of the value to store.
/// @val:	the value to store.
/// @ptr:	the pointer to store to.
///
/// Use memcpy to affect an unaligned type sized store avoiding undefined
/// behavior from approaches like type punning that require -fno-strict-aliasing
/// in order to be correct. The void* cast silences ubsan warnings.
macro_rules! __put_unaligned_t {
($type:ty, $val:expr, $ptr:expr) => {
loop {
let mut __put_unaligned_val: $type = $val;
unsafe { __builtin_memcpy($ptr, core::ptr::addr_of_mut!(__put_unaligned_val) as *const (), core::mem::size_of::<$type>() as u64) };
if true {
break;
}
}
};
}

/// __release_shared() - function to release context lock shared
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x with shared
/// access.
macro_rules! __release_shared {
($x:expr) => {
__release_shared_ctx_lock__6($x);
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask {
($y:expr) => {
$y.wrapping_sub(1)
};
}

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask__2 {
($y:expr) => {
$y.wrapping_sub(1) as u32
};
}

/// Are two types/vars the same type (ignoring qualifiers)?
macro_rules! __same_type {
() => {
0i32
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
macro_rules! __sign_use__2 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
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

macro_rules! __types_ok3__2 {
($ux:expr, $uy:expr, $uz:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy) & __sign_use__2!($uz)
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
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

macro_rules! alloc_hooks__2 {
($_do_alloc:expr) => {
{
let mut __v: *mut mempool;
let mut __v2: *mut mempool;
let mut __v3: *mut mempool = unsafe { core::mem::zeroed() };
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

macro_rules! arch_atomic_sub_return {
($i:expr, $v:expr) => {
arch_atomic_add_return($i.wrapping_neg(), $v)
};
}

/// check_add_overflow() - Calculate addition with overflow checking
/// @a: first addend
/// @b: second addend
/// @d: pointer to store sum
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted addition, regardless of whether
/// wrap-around occurred.
macro_rules! check_add_overflow__2 {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_add(__t2);
unsafe {
*$d = __t3 as i32;
}
__t.checked_add(__t2).is_none() || ((__t3 as i32) as i128) != __t3
})
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

/// check_mul_overflow() - Calculate multiplication with overflow checking
/// @a: first factor
/// @b: second factor
/// @d: pointer to store product
///
/// Returns true on wrap-around, false otherwise.
///
/// *@d holds the results of the attempted multiplication, regardless of whether
/// wrap-around occurred.
macro_rules! check_mul_overflow__2 {
($a:expr, $b:expr, $d:expr) => {
__must_check_overflow({
let __t: i128 = ($a) as i128;
let __t2: i128 = ($b) as i128;
let __t3: i128 = __t.wrapping_mul(__t2);
unsafe {
*$d = __t3 as i32;
}
__t.checked_mul(__t2).is_none() || ((__t3 as i32) as i128) != __t3
})
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

/// ilog2 - log base 2 of 32-bit or a 64-bit unsigned value
/// @n: parameter
///
/// constant-capable log of base 2 calculation
/// - this can be used to initialise global variables from constant data, hence
/// the massive ternary operator construction
///
/// selects the appropriately-sized optimised version depending on sizeof(n)
macro_rules! ilog2 {
($n:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub((($n) as u64).leading_zeros() as i32) } } else { if (core::mem::size_of::<u32>() as u64) <= 4 { __ilog2_u32($n) } else { __ilog2_u64(($n) as u64) } }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
(1i32.wrapping_neg() as u32) < 1
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

macro_rules! mempool_create {
($_min_nr:expr, $_alloc_fn:expr, $_free_fn:expr, $_pool_data:expr) => {
alloc_hooks__2!(unsafe { mempool_create_node_noprof($_min_nr, $_alloc_fn, $_free_fn, $_pool_data, 3264u32, 1i32.wrapping_neg()) })
};
}

macro_rules! mempool_create_page_pool {
($_min_nr:expr, $_order:expr) => {
mempool_create!($_min_nr, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>>(mempool_alloc_pages as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ())>>(mempool_free_pages as *const ()) }, (($_order) as i64) as *mut ())
};
}

macro_rules! mempool_create_page_pool__2 {
($_min_nr:expr) => {
mempool_create!($_min_nr, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>>(mempool_alloc_pages as *const ()) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut ())>>(mempool_free_pages as *const ()) }, core::ptr::null_mut::<()>())
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

macro_rules! offsetof {
() => {
40u64
};
}

macro_rules! offsetof__33 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, disk_dev) as u64
};
}

macro_rules! page_address {
($page:expr) => {
lowmem_page_address($page)
};
}

macro_rules! page_address__2 {
($page:expr) => {
lowmem_page_address(($page) as *const page)
};
}

macro_rules! page_to_virt {
($x:expr) => {
__va!(PFN_PHYS!(((unsafe { $x.offset_from((vmemmap_base as *mut page) as *const page) }) as i64) as u64))
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

/// round_down - round down to next specified power of 2
/// @x: the value to round
/// @y: multiple to round down to (must be a power of 2)
///
/// Rounds @x down to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding down, use rounddown() below.
macro_rules! round_down {
($x:expr, $y:expr) => {
($x) & !__round_mask__2!($y)
};
}

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask!($y)).wrapping_add(1)
};
}

/// rounddown_pow_of_two - round the given value down to nearest power of two
/// @n: parameter
///
/// round the given value down to the nearest power of two
/// - the result is undefined when n == 0
/// - this can be used to initialise global variables from constant data
macro_rules! rounddown_pow_of_two {
($n:expr) => {
if unsafe { __slate_intrinsic_1c5364fcd9da51fe($n) } { 1u64 << ilog2!($n) } else { __rounddown_pow_of_two(($n) as u64) }
};
}

macro_rules! rq_data_dir {
($rq:expr) => {
if op_is_write(req_op(($rq) as *const request) as u32) { 1i32 } else { 0i32 }
};
}

macro_rules! scsi_register_driver {
($drv:expr) => {
unsafe { __scsi_register_driver($drv, core::ptr::null_mut::<module>()) }
};
}

macro_rules! scsi_unregister_driver {
($drv:expr) => {
unsafe { driver_unregister(core::ptr::addr_of_mut!((*$drv).gendrv)) };
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

/// Use these to separate status msg and our bytes
///
/// These are set by:
///
///     status byte = set from target device
///     msg_byte    (unused)
///     host_byte   = set by low-level driver to indicate status.
macro_rules! status_byte {
($result:expr) => {
($result) & 255
};
}

/// sysfs_match_string - matches given string in an array
/// @_a: array of strings
/// @_s: string to match with
///
/// Helper for __sysfs_match_string(). Calculates the size of @a automatically.
macro_rules! sysfs_match_string {
($_a:expr, $_s:expr) => {
unsafe { __sysfs_match_string(($_a) as *const *const u8, ARRAY_SIZE__5!(), $_s) }
};
}

/// sysfs_match_string - matches given string in an array
/// @_a: array of strings
/// @_s: string to match with
///
/// Helper for __sysfs_match_string(). Calculates the size of @a automatically.
macro_rules! sysfs_match_string__2 {
($_a:expr, $_s:expr) => {
unsafe { __sysfs_match_string(($_a) as *const *const u8, ARRAY_SIZE__26!(), $_s) }
};
}

/// sysfs_match_string - matches given string in an array
/// @_a: array of strings
/// @_s: string to match with
///
/// Helper for __sysfs_match_string(). Calculates the size of @a automatically.
macro_rules! sysfs_match_string__3 {
($_a:expr, $_s:expr) => {
unsafe { __sysfs_match_string(($_a) as *const *const u8, ARRAY_SIZE__2!(), $_s) }
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

/// time_after_eq - returns true if the time a is after or the same as time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is after or the same as time b, otherwise %false.
macro_rules! time_after_eq {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__3!(u64) != 0 && typecheck__4!(u64) != 0;
__v && ($a.wrapping_sub($b) as i64) >= 0
}
};
}

/// time_before_eq - returns true if the time a is before or the same as time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is before or the same as time b, otherwise %false.
macro_rules! time_before_eq {
($a:expr, $b:expr) => {
time_after_eq!($b, $a)
};
}

macro_rules! to_scsi_disk {
($obj:expr) => {
{
let mut __v: *mut scsi_disk;
__v = ((unsafe { ((($obj) as *mut ()) as *const u8).wrapping_offset(-(offsetof__33!(scsi_disk) as isize)) }) as *mut ()) as *mut scsi_disk;
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
macro_rules! typecheck__3 {
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
macro_rules! typecheck__4 {
($type:ty) => {
{
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == (core::ptr::addr_of_mut!(__v3) as *mut $type);
__v = 1;
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

unsafe extern "C" {
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut kmsg_fops: file_operations;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut system_state: u32;
    static mut cpuinfo_op: seq_operations;
    static mut RCU: *const __ctx_lock_RCU;
    static mut jiffies: u64;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn abort();
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __xadd_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memmove(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn strlen(arg0: *const u8) -> u64;
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn __sysfs_match_string(arg0: *const *const u8, arg1: u64, arg2: *const u8) -> i32;
    fn _kstrtoul(arg0: *const u8, arg1: u32, arg2: *mut u64) -> i32;
    fn kstrtoull(arg0: *const u8, arg1: u32, arg2: *mut u64) -> i32;
    fn kstrtouint(arg0: *const u8, arg1: u32, arg2: *mut u32) -> i32;
    fn kstrtoint(arg0: *const u8, arg1: u32, arg2: *mut i32) -> i32;
    fn kstrtobool(arg0: *const u8, arg1: *mut bool) -> i32;
    fn sprintf(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn mempool_create_node_noprof(arg0: i32, arg1: Option<unsafe extern "C" fn(u32, *mut ()) -> *mut ()>, arg2: Option<unsafe extern "C" fn(*mut (), *mut ())>, arg3: *mut (), arg4: u32, arg5: i32) -> *mut mempool;
    fn mempool_destroy(arg0: *mut mempool);
    fn mempool_alloc_noprof(arg0: *mut mempool, arg1: u32) -> *mut ();
    fn mempool_free(arg0: *mut (), arg1: *mut mempool);
    fn mempool_alloc_pages(arg0: u32, arg1: *mut ()) -> *mut ();
    fn mempool_free_pages(arg0: *mut (), arg1: *mut ());
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn capable(arg0: i32) -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_360();
    fn __compiletime_assert_385();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_391();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn ida_alloc_range(arg0: *mut ida, arg1: u32, arg2: u32, arg3: u32) -> i32;
    fn ida_free(arg0: *mut ida, arg1: u32);
    fn sysfs_emit(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn __compiletime_assert_488();
    fn class_register(arg0: *const class) -> i32;
    fn class_unregister(arg0: *const class);
    fn driver_unregister(arg0: *mut device_driver);
    fn dev_set_name(arg0: *mut device, arg1: *const u8, ...) -> i32;
    fn device_unregister(arg0: *mut device);
    fn device_initialize(arg0: *mut device);
    fn device_add(arg0: *mut device) -> i32;
    fn device_del(arg0: *mut device);
    fn get_device(arg0: *mut device) -> *mut device;
    fn put_device(arg0: *mut device);
    fn device_add_disk(arg0: *mut device, arg1: *mut gendisk, arg2: *mut *const attribute_group) -> i32;
    fn del_gendisk(arg0: *mut gendisk);
    fn set_disk_ro(arg0: *mut gendisk, arg1: bool);
    fn set_capacity_and_notify(arg0: *mut gendisk, arg1: u64) -> bool;
    fn put_disk(arg0: *mut gendisk);
    fn __register_blkdev(arg0: u32, arg1: *const u8, arg2: Option<unsafe extern "C" fn(u32)>) -> i32;
    fn unregister_blkdev(arg0: u32, arg1: *const u8);
    fn disk_check_media_change(arg0: *mut gendisk) -> bool;
    fn queue_limits_commit_update_frozen(arg0: *mut request_queue, arg1: *mut queue_limits) -> i32;
    fn blk_queue_rq_timeout(arg0: *mut request_queue, arg1: u32);
    fn disk_alloc_independent_access_ranges(arg0: *mut gendisk, arg1: i32) -> *mut blk_independent_access_ranges;
    fn disk_set_independent_access_ranges(arg0: *mut gendisk, arg1: *mut blk_independent_access_ranges);
    fn blkdev_compat_ptr_ioctl(arg0: *mut block_device, arg1: u32, arg2: u32, arg3: u64) -> i32;
    fn blk_pm_runtime_init(arg0: *mut request_queue, arg1: *mut device);
    fn msleep(arg0: u32);
    fn string_get_size(arg0: u64, arg1: u64, arg2: u32, arg3: *mut u8, arg4: i32) -> i32;
    fn __pm_runtime_resume(arg0: *mut device, arg1: i32) -> i32;
    fn pm_runtime_set_autosuspend_delay(arg0: *mut device, arg1: i32);
    fn blk_mq_alloc_disk_for_queue(arg0: *mut request_queue, arg1: *mut lock_class_key) -> *mut gendisk;
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn block_pr_type_to_scsi(arg0: u32) -> u32;
    fn scsi_pr_type_to_block(arg0: u32) -> u32;
    fn sdev_prefix_printk(arg0: *const u8, arg1: *const scsi_device, arg2: *const u8, arg3: *const u8, ...);
    fn scmd_printk(arg0: *const u8, arg1: *mut scsi_cmnd, arg2: *const u8, ...);
    fn scsi_device_get(arg0: *mut scsi_device) -> i32;
    fn scsi_device_put(arg0: *mut scsi_device);
    fn scsi_set_medium_removal(arg0: *mut scsi_device, arg1: u8) -> i32;
    fn scsi_mode_sense(arg0: *mut scsi_device, arg1: i32, arg2: i32, arg3: i32, arg4: *mut u8, arg5: i32, arg6: i32, arg7: i32, arg8: *mut scsi_mode_data, arg9: *mut scsi_sense_hdr) -> i32;
    fn scsi_mode_select(arg0: *mut scsi_device, arg1: i32, arg2: i32, arg3: *mut u8, arg4: i32, arg5: i32, arg6: i32, arg7: *mut scsi_mode_data, arg8: *mut scsi_sense_hdr) -> i32;
    fn scsi_test_unit_ready(arg0: *mut scsi_device, arg1: i32, arg2: i32, arg3: *mut scsi_sense_hdr) -> i32;
    fn scsi_get_vpd_page(arg0: *mut scsi_device, arg1: u8, arg2: *mut u8, arg3: i32) -> i32;
    fn scsi_report_opcode(arg0: *mut scsi_device, arg1: *mut u8, arg2: u32, arg3: u8, arg4: u16) -> i32;
    fn scsi_device_set_state(arg0: *mut scsi_device, arg1: u32) -> i32;
    fn scsi_execute_cmd(arg0: *mut scsi_device, arg1: *const u8, arg2: u32, arg3: *mut (), arg4: u32, arg5: i32, arg6: i32, arg7: *const scsi_exec_args) -> i32;
    fn scsi_failures_reset_retries(arg0: *mut scsi_failures);
    fn scsi_autopm_get_device(arg0: *mut scsi_device) -> i32;
    fn scsi_autopm_put_device(arg0: *mut scsi_device);
    fn scsi_alloc_sgtables(arg0: *mut scsi_cmnd) -> u8;
    fn scsi_free_sgtables(arg0: *mut scsi_cmnd);
    fn scsi_print_command(arg0: *mut scsi_cmnd);
    fn scsi_print_sense_hdr(arg0: *const scsi_device, arg1: *const u8, arg2: *const scsi_sense_hdr);
    fn scsi_hostbyte_string(arg0: i32) -> *const u8;
    fn __scsi_register_driver(arg0: *mut scsi_driver, arg1: *mut module) -> i32;
    fn scsi_block_when_processing_errors(arg0: *mut scsi_device) -> i32;
    fn scsi_command_normalize_sense(arg0: *const scsi_cmnd, arg1: *mut scsi_sense_hdr) -> bool;
    fn scsi_get_sense_info_fld(arg0: *const u8, arg1: i32, arg2: *mut u64) -> bool;
    fn scsi_ioctl_block_when_processing_errors(arg0: *mut scsi_device, arg1: i32, arg2: bool) -> i32;
    fn scsi_ioctl(arg0: *mut scsi_device, arg1: bool, arg2: i32, arg3: *mut ()) -> i32;
    fn scsicam_bios_param(arg0: *mut gendisk, arg1: u64, arg2: *mut i32) -> i32;
    fn sd_dif_config_host(arg0: *mut scsi_disk, arg1: *mut queue_limits);
    fn __compiletime_assert_619();
    fn __compiletime_assert_622();
    fn __compiletime_assert_625();
    fn __compiletime_assert_628();
    fn __compiletime_assert_631();
    fn __compiletime_assert_637();
    fn __compiletime_assert_640();
    fn __compiletime_assert_644();
    fn __compiletime_assert_650();
    fn __compiletime_assert_653();
    fn __compiletime_assert_656();
    fn __compiletime_assert_659();
    fn __compiletime_assert_661();
    fn __compiletime_assert_663();
    fn __compiletime_assert_665();
    fn __compiletime_assert_667();
    fn __compiletime_assert_670();
    fn __compiletime_assert_673();
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
struct io_bitmap {
}


#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely!(overflow) != 0
}

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


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
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
fn __acquire_shared_ctx_lock__1(var: &raw_spinlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__1(var: &raw_spinlock) {
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


fn kmsan_unpoison_memory(address: *const (), size: u64) {
}

fn clear_pages(mut addr: *mut (), npages: u32) {
let mut len: u64 = (npages as u64).wrapping_mul(_AC!() << 12i32);
// Clean up KMSAN metadata for the pages being cleared. The assembly call
// below clobbers @addr, so perform unpoisoning before it.
kmsan_unpoison_memory(addr as *const (), len);
// The inline asm embeds a CALL instruction and usually that is a no-no
// due to the compiler not knowing that and thus being unable to track
// callee-clobbered registers.
//
// In this case that is fine because the registers clobbered by
// __clear_pages_unrolled() are part of the inline asm register
// specification.
unsafe {
let __t0: u64;
let __t1: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\tcall __clear_pages_unrolled\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 3*32+16)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tshrq $3, %rcx; rep stosq\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 9*32+ 9)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\trep stosb\n775:\n.popsection\n", inlateout("cx") len as u64 => __t0, inlateout("di") addr as u64 => __t1, in("ax") (0 as i32) as u32, options(att_syntax));
len = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
addr = unsafe { core::mem::transmute::<u64, *mut ()>(__t1 as u64) };
}
}

fn clear_page(addr: *mut ()) {
clear_pages(addr, 1u32);
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
#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
}


// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
fn fls_long(l: u64) -> u32 {
if ((const { core::mem::size_of::<u64>() }) as u64) == 4 {
return fls(l as u32) as u32;
}
fls64(l) as u32
}

// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
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
struct pollfd {
}


#[inline(always)]
fn is_power_of_2(n: u64) -> bool {
n.wrapping_sub(1) < n ^ n.wrapping_sub(1)
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


fn __rounddown_pow_of_two(n: u64) -> u64 {
1u64 << fls_long(n).wrapping_sub(1)
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
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
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pgtable_types_h_295 {
    pgd: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
    __slate_bits_0: __slate_bits::__SlateBits115U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
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
    __slate_bits_0: __slate_bits::__SlateBits120U0,
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
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

#[inline(always)]
fn raw_atomic_sub_return(i: i32, v: *mut __SlateRecord_types_h_190) -> i32 {
arch_atomic_sub_return!(i, v)
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
fn raw_atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
raw_atomic_sub_return(1i32, v)
}

#[inline(always)]
fn raw_atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
arch_atomic_dec_and_test(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
fn atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_return(v)
}

#[inline(always)]
fn atomic_dec_and_test(v: *mut __SlateRecord_types_h_190) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_and_test(v)
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


fn kstrtoul(s: *const u8, base: u32, res: *mut u64) -> i32 {
// We want to shortcut function call, but
// __builtin_types_compatible_p(unsigned long, unsigned long long) = 0.
if ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<u64>() }) as u64) && ((const { core::mem::align_of::<u64>() }) as u64) == ((const { core::mem::align_of::<u64>() }) as u64) {
return unsafe { kstrtoull(s, base, res as *mut u64) };
} else {
return unsafe { _kstrtoul(s, base, res) };
}
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
    __slate_bits_0: __slate_bits::__SlateBits121U0,
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
#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[inline(always)]
fn __acquire_shared_ctx_lock__2(var: &spinlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__2(var: &spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[inline(always)]
fn __acquire_shared_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__3(var: &rwlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__4(var: &mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn __release_shared_ctx_lock__4(var: &mutex) {
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
struct mutex {
    owner: __SlateRecord_types_h_197,
    wait_lock: raw_spinlock,
    osq: optimistic_spin_queue,
    first_waiter: *mut mutex_waiter,
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
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct audit_context {
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
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct rq_flags {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
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
struct vm_struct {
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
struct kmem_cache {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
struct mem_cgroup {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__5(var: &seqlock) {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
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
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
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
struct uprobe_xol_ops {
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
    __slate_bits_0: __slate_bits::__SlateBits277U0,
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


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct codetag {
    flags: u32,
    lineno: u32,
    modname: *const u8,
    function: *const u8,
    filename: *const u8,
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


fn mem_alloc_profiling_enabled() -> bool {
false
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
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
struct shrink_control {
    gfp_mask: u32,
    nid: i32,
    nr_to_scan: u64,
    nr_scanned: u64,
    memcg: *mut mem_cgroup,
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
struct kref {
    refcount: refcount_struct,
}


#[inline(always)]
fn __acquire_shared_ctx_lock__6(var: *const __ctx_lock_RCU) {
}

#[inline(always)]
fn __release_shared_ctx_lock__6(var: *const __ctx_lock_RCU) {
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
struct __ctx_lock_RCU {
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
fn rcu_read_lock_held() -> i32 {
1
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
#[inline(always)]
fn rcu_read_lock() {
unsafe { __rcu_read_lock() };
__acquire_shared!(unsafe { RCU });
rcu_lock_acquire!();
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
}

fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!(unsafe { RCU });
unsafe { __rcu_read_unlock() };
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


#[inline(always)]
fn __acquire_shared_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __release_shared_ctx_lock__7(var: &rw_semaphore) {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__8(var: &srcu_struct) {
}

#[inline(always)]
fn __release_shared_ctx_lock__8(var: &srcu_struct) {
}

// Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
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
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
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
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
    __slate_bits_0: __slate_bits::__SlateBits412U0,
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
fn PageHighMem(page: *const page) -> i32 {
0
}

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
fn __acquire_shared_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct backing_dev_info {
}


#[inline(always)]
fn __release_shared_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_lock {
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
fn __acquire_shared_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
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
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
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
struct list_lru {
    node: *mut list_lru_node,
}


#[inline(always)]
fn __acquire_shared_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__11(var: &__context_bitlock) {
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
struct __context_bitlock {
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
struct fiemap_extent_info {
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
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
struct group_info {
    usage: refcount_struct,
    ngroups: i32,
    gid: [__SlateRecord_uidgid_types_h_13; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct cpu_itimer {
    expires: u64,
    incr: u64,
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
struct task_cputime_atomic {
    utime: __SlateRecord_types_h_197,
    stime: __SlateRecord_types_h_197,
    sum_exec_runtime: __SlateRecord_types_h_197,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
    __slate_bits_0: __slate_bits::__SlateBits280U0,
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
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


// group exit in progress
fn task_rlimit(task: *const task_struct, limit: u32) -> u64 {
let mut __v0: u64;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_360() };
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
struct sock {
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
struct wake_irq {
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
struct dev_pm_qos {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
unsafe { __compiletime_assert_385() };
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
    __slate_bits_0: __slate_bits::__SlateBits562U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct module_param_attrs {
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
unsafe { __compiletime_assert_391() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
} else {
loop {
unsafe {
core::arch::asm!("392: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 392b - ., 3; .popsection\n# {0}", const 392 as i32, options(nostack, att_syntax));
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_file_ref_h_52 {
    refcnt: __SlateRecord_types_h_197,
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
union __SlateRecord_device_h_108 {
    show: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *mut u8) -> i64>,
    show_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *mut u8) -> i64>,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_device_h_114 {
    store: Option<unsafe extern "C" fn(*mut device, *mut device_attribute, *const u8, u64) -> i64>,
    store_const: Option<unsafe extern "C" fn(*mut device, *const device_attribute, *const u8, u64) -> i64>,
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
struct disk_stats {
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
struct bio_alloc_cache {
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
struct rq_qos {
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
struct timer_rand_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cdrom_device_info {
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


// Set the IDR flag and the IDR_FREE tag
#[repr(C)]
#[derive(Clone, Copy)]
struct ida {
    xa: xarray,
}


fn ida_alloc(ida: *mut ida, gfp: u32) -> i32 {
unsafe { ida_alloc_range(ida, 0u32, 4294967295u32, gfp) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct blk_mq_ctx {
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
struct opal_dev {
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
    __slate_bits_0: __slate_bits::__SlateBits634U0,
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
    __slate_bits_0: __slate_bits::__SlateBits574U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
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
struct bsg_device {
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
fn kmap_local_page(page: *const page) -> *mut () {
page_address!(page)
}

fn __kunmap_local(addr: *const ()) {
}

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
struct scsi_device_handler {
}


fn clear_highpage(page: *mut page) {
let kaddr: *mut () = kmap_local_page(page as *const page);
clear_page(kaddr);
loop {
loop {
if __same_type!() != 0 {
unsafe { __compiletime_assert_488() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__kunmap_local(kaddr as *const ());
if true {
break;
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


fn bvec_set_page(bv: *mut bio_vec, page: *mut page, len: u32, offset: u32) {
unsafe {
(*bv).bv_page = page;
}
unsafe {
(*bv).bv_len = len;
}
unsafe {
(*bv).bv_offset = offset;
}
}

#[repr(C, packed(4))]
#[derive(Clone, Copy)]
struct bvec_iter {
    bi_sector: u64,
    bi_size: u32,
    bi_idx: u32,
    bi_offset: u32,
}


fn bvec_virt(bvec: *mut bio_vec) -> *mut () {
let mut __v0: i64;
let __ret_warn_on: i32 = (PageHighMem((unsafe { (*bvec).bv_page }) as *const page) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("531: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 531b - ., 3; .popsection\n# {0}", const 531 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_bvec_virt_1, sym __slate_asm_string_bvec_virt_2, const 333 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("532: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 532b - ., 4; .popsection\n# {0}", const 532 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
(unsafe { (page_address__2!(unsafe { (*bvec).bv_page }) as *const u8).wrapping_offset((*bvec).bv_offset as isize) }) as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_transport_template {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pm_message {
    event: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_scsi_host_h_909 {
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
struct __SlateRecord_sd_c_246 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_309 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_309_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_309_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_309_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_309_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_309_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_340 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_340_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_340_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_340_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_340_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_340_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_371 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_371_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_371_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_371_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_371_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_371_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_400 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_400_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_400_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_400_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_400_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_400_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_429 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_429_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_429_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_429_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_429_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_429_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_460 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_460_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_460_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_460_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_460_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_460_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_470 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_470_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_470_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_470_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_470_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_470_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_479 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_479_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_479_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_479_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_479_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_479_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_511 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_511_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_511_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_511_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_511_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_511_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_534 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_534_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_534_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_534_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_534_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_534_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_543 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_543_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_543_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_543_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_543_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_543_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_553 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_553_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_553_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_553_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_553_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_553_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_589 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_600 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_600_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_600_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_600_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_600_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_600_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_629 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_637 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_637_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_637_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_637_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_637_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_637_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_663 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_663_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_663_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_663_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_663_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_663_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_709 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_709_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_709_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_709_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_709_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_709_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_724 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_724_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_724_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_724_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_724_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_724_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_757 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_757_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_757_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_757_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_757_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_757_6 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_sd_c_3288 {
    h: scsi_stream_status_header,
    s: scsi_stream_status,
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
    pub struct __SlateBits277U0 {
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
    pub struct __SlateBits562U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits662U0 {
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
    pub struct __SlateBits659U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits662U1 {
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
    pub struct __SlateBits662U2 {
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
    pub struct __SlateBits662U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits634U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits574U0 {
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
    pub struct __SlateBits120U0 {
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
    pub struct __SlateBits412U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits280U0 {
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
    pub struct __SlateBits115U0 {
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
    pub struct __SlateBits121U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits833U0 {
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
    pub struct __SlateBits846U0 {
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
    pub struct __SlateBits832U0 {
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
    pub struct __SlateBits831U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits827U0 {
        #[bits(1)]
        pub longlba: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield([u8; 3], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits854U0 {
        #[bits(1)]
        pub ATO: u32,
        #[bits(1)]
        pub cache_override: u32,
        #[bits(1)]
        pub WCE: u32,
        #[bits(1)]
        pub RCD: u32,
        #[bits(1)]
        pub DPOFUA: u32,
        #[bits(1)]
        pub first_scan: u32,
        #[bits(1)]
        pub lbpme: u32,
        #[bits(1)]
        pub lbprz: u32,
        #[bits(1)]
        pub lbpu: u32,
        #[bits(1)]
        pub lbpws: u32,
        #[bits(1)]
        pub lbpws10: u32,
        #[bits(1)]
        pub lbpvpd: u32,
        #[bits(1)]
        pub ws10: u32,
        #[bits(1)]
        pub ws16: u32,
        #[bits(2)]
        pub rc_basis: u32,
        #[bits(2)]
        pub zoned: u32,
        #[bits(1)]
        pub urswrz: u32,
        #[bits(1)]
        pub security: u32,
        #[bits(1)]
        pub ignore_medium_access_errors: u32,
        #[bits(1)]
        pub rscs: u32,
        #[bits(1)]
        pub use_atomic_write_boundary: u32,
        #[bits(1, access = na)]
        pub __slate_pad_21: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits817U0 {
        #[bits(7)]
        pub reserved1: u8,
        #[bits(1)]
        pub perm: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits817U1 {
        #[bits(6)]
        pub rel_lifetime: u8,
        #[bits(2)]
        pub reserved3: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits816U0 {
        #[bits(1)]
        pub ic_enable: u8,
        #[bits(1)]
        pub cs_enble: u8,
        #[bits(1)]
        pub st_enble: u8,
        #[bits(3)]
        pub reserved1: u8,
        #[bits(2)]
        pub io_advice_hints_mode: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits816U1 {
        #[bits(4)]
        pub lbm_descriptor_type: u8,
        #[bits(2)]
        pub rlbsr: u8,
        #[bits(1)]
        pub reserved3: u8,
        #[bits(1)]
        pub acdlu: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits662U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits662U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits662U2,
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
    __slate_bits_3: __slate_bits::__SlateBits662U3,
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
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
}


// DEVICE_ATTR_RW - Define a read-write device attribute.
// @_name: Attribute name.
//
// Like DEVICE_ATTR(), but @_mode is 0644, @_show is <_name>_show,
// and @_store is <_name>_store.
// DEVICE_ATTR_RO - Define a readable device attribute.
// @_name: Attribute name.
//
// Like DEVICE_ATTR(), but @_mode is 0444 and @_show is <_name>_show.
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
    __slate_bits_0: __slate_bits::__SlateBits659U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
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


fn op_is_write(op: u32) -> bool {
op & 1 != 0
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


fn bio_integrity_flagged(bio: *mut bio, flag: u32) -> bool {
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hd_geometry {
    heads: u8,
    sectors: u8,
    cylinders: u16,
    start: u64,
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


// open for writing
// opened with O_NDELAY
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


fn blk_validate_block_size(bsize: u64) -> i32 {
if bsize < 512 || bsize > _AC!() << 12i32 || !is_power_of_2(bsize) {
return 22i32.wrapping_neg();
}
0
}

// supports a volatile write cache
// supports passing on the FUA bit
// rotational device (hard drive or floppy)
// contributes to the random number pool
// don't modify data until writeback is done
// always completes in submit context
// supports REQ_NOWAIT
// is a zoned device
// atomic writes enabled
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


fn bdev_partno(bdev: *const block_device) -> u8 {
(atomic_read(unsafe { core::ptr::addr_of!((*bdev).__bd_flags) }) & 255) as u8
}

fn bdev_nr_sectors(bdev: *mut block_device) -> u64 {
unsafe { (*bdev).bd_nr_sectors }
}

fn get_capacity(disk: *mut gendisk) -> u64 {
bdev_nr_sectors(unsafe { (*disk).part0 })
}

fn queue_limits_start_update(q: *mut request_queue) -> queue_limits {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*q).limits_lock)) };
return unsafe { (*q).limits };
}

fn blk_queue_disable_discard(q: *mut request_queue) {
unsafe {
(*q).limits.max_discard_sectors = 0;
}
}

fn blk_queue_disable_write_zeroes(q: *mut request_queue) {
unsafe {
(*q).limits.max_write_zeroes_sectors = 0;
}
unsafe {
(*q).limits.max_wzeroes_unmap_sectors = 0;
}
}

fn bdev_is_partition(bdev: *mut block_device) -> bool {
(bdev_partno(bdev as *const block_device) as i32) != 0
}

core::arch::global_asm!(".section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_sd_mod__674_4516_init_sd6:\t\t\t\n.long\t{0} - .\t\n.previous\t\t\t\t\t\n", sym init_sd, options(att_syntax));

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
fn free_opal_dev(dev: *mut opal_dev) {
}

fn is_sed_ioctl(cmd: u32) -> bool {
false
}

fn sed_ioctl(dev: *mut opal_dev, cmd: u32, ioctl_ptr: *mut ()) -> i32 {
0
}

fn opal_unlock_from_suspend(dev: *mut opal_dev) -> bool {
false
}

// Request is asynchronous
// Increment/decrement the
// usage_count
// Use autosuspend_delay
// Succeed if runtime PM is disabled
fn pm_runtime_suspended(dev: *mut device) -> bool {
((unsafe { (*dev).power.runtime_status }) as i32) == 2 && ((unsafe { (*dev).power.__slate_bits_2.__get_disable_depth() }) as i32) == 0
}

fn pm_request_resume(dev: *mut device) -> i32 {
unsafe { __pm_runtime_resume(dev, 1i32) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pr_keys {
    generation: u32,
    num_keys: u32,
    keys: [u64; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_held_reservation {
    key: u64,
    generation: u32,
    r#type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pr_ops {
    pr_register: Option<unsafe extern "C" fn(*mut block_device, u64, u64, u32) -> i32>,
    pr_reserve: Option<unsafe extern "C" fn(*mut block_device, u64, u32, u32) -> i32>,
    pr_release: Option<unsafe extern "C" fn(*mut block_device, u64, u32) -> i32>,
    pr_preempt: Option<unsafe extern "C" fn(*mut block_device, u64, u64, u32, bool) -> i32>,
    pr_clear: Option<unsafe extern "C" fn(*mut block_device, u64) -> i32>,
    pr_read_keys: Option<unsafe extern "C" fn(*mut block_device, *mut pr_keys) -> i32>,
    pr_read_reservation: Option<unsafe extern "C" fn(*mut block_device, *mut pr_held_reservation) -> i32>,
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


fn req_op(req: *const request) -> u32 {
((unsafe { (*req).cmd_flags }) & 255u32) as u32
}

fn req_get_ioprio(req: *mut request) -> u16 {
if !unsafe { (*req).bio }.is_null() {
return unsafe { (*(*req).bio).bi_ioprio };
}
0
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


fn blk_mq_rq_from_pdu(pdu: *mut ()) -> *mut request {
((unsafe { (pdu as *const u8).wrapping_offset(-(core::mem::size_of::<request>() as isize)) }) as *mut ()) as *mut request
}

fn blk_rq_pos(rq: *const request) -> u64 {
unsafe { (*rq).__sector }
}

fn blk_rq_bytes(rq: *const request) -> u32 {
unsafe { (*rq).__data_len }
}

fn blk_rq_sectors(rq: *const request) -> u32 {
blk_rq_bytes(rq) >> 9i32
}

fn get_unaligned_be16(p: *const ()) -> u16 {
__swab16!(__get_unaligned_t__2!(u16, p))
}

fn get_unaligned_be32(p: *const ()) -> u32 {
__swab32!(__get_unaligned_t!(u32, p))
}

fn get_unaligned_be64(p: *const ()) -> u64 {
__swab64!(__get_unaligned_t__3!(u64, p))
}

fn put_unaligned_be16(val: u16, p: *mut ()) {
__put_unaligned_t!(u16, __swab16!(val), p);
}

fn put_unaligned_be32(val: u32, p: *mut ()) {
__put_unaligned_t!(u32, __swab32!(val), p);
}

fn put_unaligned_be64(val: u64, p: *mut ()) {
__put_unaligned_t!(u64, __swab64!(val), p);
}

// values for ZBC_OUT
// values for PR in service action
// SENSE KEYS
// Magneto-optical disk -
// - treated as TYPE_DISK
#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_io_group_descriptor {
    __slate_bits_0: __slate_bits::__SlateBits816U0,
    reserved2: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits816U1,
    params: [u8; 2],
    reserved4: u8,
    reserved5: [u8; 8],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_stream_status {
    __slate_bits_0: __slate_bits::__SlateBits817U0,
    reserved2: u8,
    stream_identifier: u16,
    __slate_bits_1: __slate_bits::__SlateBits817U1,
    reserved4: [u8; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_stream_status_header {
    len: u32,
    reserved: u16,
    number_of_open_streams: u16,
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


fn scsi_sense_valid(sshdr: *const scsi_sense_hdr) -> bool {
if sshdr.is_null() {
return false;
}
((unsafe { (*sshdr).response_code }) as i32) & 112 == 112
}

fn scsi_status_is_check_condition(mut status: i32) -> i32 {
if status < 0 {
return 0;
}
status &= 254;
(status == 2) as i32
}

fn scsi_status_is_good(mut status: i32) -> bool {
if status < 0 {
return false;
}
if host_byte!(status) == 1 {
return false;
}
// FIXME: bit0 is listed as reserved in SCSI-2, but is
// significant in SCSI-3.  For now, we follow the SCSI-2
// behaviour and ignore reserved bits.
status &= 254;
return status == 0 || status == 4 || status == 16 || status == 20 || status == 34; // Next two "intermediate" statuses are obsolete in SAM-4
// FIXME: this is obsolete in SAM-3
}

#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_mode_data {
    length: u32,
    block_descriptor_length: u16,
    medium_type: u8,
    device_specific: u8,
    header_length: u8,
    __slate_bits_0: __slate_bits::__SlateBits827U0,
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
    __slate_bits_0: __slate_bits::__SlateBits831U0,
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
    __slate_bits_0: __slate_bits::__SlateBits833U0,
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

// scsi_execute_cmd users can set scsi_failure.result to have
// scsi_check_passthrough fail/retry a command. scsi_failure.result can be a
// specific host byte or message code, or SCMD_FAILURE_RESULT_ANY can be used
// to match any host or message code.
// Set scsi_failure.result to SCMD_FAILURE_STAT_ANY to fail/retry any failure
// scsi_status_is_good returns false for.
#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_failure {
    result: i32,
    sense: u8,
    asc: u8,
    ascq: u8,
    allowed: i8,
    retries: i8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_failures {
    total_allowed: i32,
    total_retries: i32,
    failure_definitions: *mut scsi_failure,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_exec_args {
    sense: *mut u8,
    sense_len: u32,
    sshdr: *mut scsi_sense_hdr,
    req_flags: u32,
    scmd_flags: i32,
    resid: *mut i32,
    failures: *mut scsi_failures,
}


fn scsi_device_online(sdev: *mut scsi_device) -> i32 {
(((unsafe { (*sdev).sdev_state }) as u32) != 6 && ((unsafe { (*sdev).sdev_state }) as u32) != 7 && ((unsafe { (*sdev).sdev_state }) as u32) != 4) as i32
}

fn scsi_device_protection(sdev: *mut scsi_device) -> i32 {
if ((unsafe { (*sdev).__slate_bits_0.__get_no_dif() }) as i32) != 0 {
return 0;
}
(((unsafe { (*sdev).scsi_level }) as i32) > 3 && ((unsafe { *unsafe { (*sdev).inquiry }.wrapping_offset(5 as isize) }) as i32) & 1i32 != 0) as i32
}

fn scsi_device_supports_vpd(sdev: *mut scsi_device) -> i32 {
// Attempt VPD inquiry if the device blacklist explicitly calls
// for it.
if ((unsafe { (*sdev).__slate_bits_0.__get_try_vpd_pages() }) as i32) != 0 {
return 1;
}
// Although VPD inquiries can go to SCSI-2 type devices,
// some USB ones crash on receiving them, and the pages
// we currently ask for are mandatory for SPC-2 and beyond
if ((unsafe { (*sdev).scsi_level }) as i32) >= 5 && ((unsafe { (*sdev).__slate_bits_0.__get_skip_vpd_pages() }) as i32) == 0 {
return 1;
}
0
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


fn scsi_cmd_to_rq(scmd: *mut scsi_cmnd) -> *mut request {
blk_mq_rq_from_pdu(scmd as *mut ())
}

fn scsi_bufflen(cmd: *mut scsi_cmnd) -> u32 {
unsafe { (*cmd).sdb.length }
}

fn scsi_set_resid(cmd: *mut scsi_cmnd, resid: u32) {
unsafe {
(*cmd).resid_len = resid;
}
}

fn scsi_get_resid(cmd: *mut scsi_cmnd) -> u32 {
unsafe { (*cmd).resid_len }
}

fn scsi_set_prot_op(scmd: *mut scsi_cmnd, op: u8) {
unsafe {
(*scmd).prot_op = op;
}
}

fn scsi_set_prot_type(scmd: *mut scsi_cmnd, r#type: u8) {
unsafe {
(*scmd).prot_type = r#type;
}
}

fn scsi_prot_sg_count(cmd: *mut scsi_cmnd) -> u32 {
if !unsafe { (*cmd).prot_sdb }.is_null() { unsafe { (*(*cmd).prot_sdb).table.nents } } else { 0 }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_driver {
    gendrv: device_driver,
    probe: Option<unsafe extern "C" fn(*mut scsi_device) -> i32>,
    remove: Option<unsafe extern "C" fn(*mut scsi_device)>,
    shutdown: Option<unsafe extern "C" fn(*mut scsi_device)>,
    resume: Option<unsafe extern "C" fn(*mut device) -> i32>,
    rescan: Option<unsafe extern "C" fn(*mut device)>,
    init_command: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> u8>,
    uninit_command: Option<unsafe extern "C" fn(*mut scsi_cmnd)>,
    done: Option<unsafe extern "C" fn(*mut scsi_cmnd) -> i32>,
    eh_action: Option<unsafe extern "C" fn(*mut scsi_cmnd, i32) -> i32>,
    eh_reset: Option<unsafe extern "C" fn(*mut scsi_cmnd)>,
}


fn scsi_sense_is_deferred(sshdr: &scsi_sense_hdr) -> bool {
((unsafe { (*(sshdr as *const scsi_sense_hdr)).response_code }) as i32) >= 112 && ((unsafe { (*(sshdr as *const scsi_sense_hdr)).response_code }) as i32) & 1 != 0
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
    __slate_bits_0: __slate_bits::__SlateBits846U0,
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
    __slate_bits_0: __slate_bits::__SlateBits832U0,
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


fn scsi_host_dif_capable(shost: *mut Scsi_Host, target_type: u32) -> u32 {
if (target_type as u64) >= ARRAY_SIZE__12!() {
return 0;
}
if (unsafe { (*shost).prot_capabilities }) & (((unsafe { cap[target_type as usize] }) as i32) as u32) != 0 { target_type } else { 0 }
}

static mut cap: [u8; 4] = [0, 1, 2, 4];

fn scsi_host_dix_capable(shost: *mut Scsi_Host, target_type: u32) -> u32 {
0
}

// Time out in seconds for disks and Magneto-opticals (which are slower).
// Flush timeout is a multiplier over the standard device timeout which is
// user modifiable via sysfs but initially set to SD_TIMEOUT
// Number of allowed retries
// Size of the initial data buffer for mode and read capacity data
// Number of sectors at the end of the device to avoid multi-sector
// accesses to in the case of last_sector_bug
#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_disk {
    device: *mut scsi_device,
    disk_dev: device,
    disk: *mut gendisk,
    opal_dev: *mut opal_dev,
    openers: __SlateRecord_types_h_190,
    capacity: u64,
    max_retries: i32,
    min_xfer_blocks: u32,
    max_xfer_blocks: u32,
    opt_xfer_blocks: u32,
    max_ws_blocks: u32,
    max_unmap_blocks: u32,
    unmap_granularity: u32,
    unmap_alignment: u32,
    max_atomic: u32,
    atomic_alignment: u32,
    atomic_granularity: u32,
    max_atomic_with_boundary: u32,
    max_atomic_boundary: u32,
    index: u32,
    physical_block_size: u32,
    max_medium_access_timeouts: u32,
    medium_access_timed_out: u32,
    permanent_stream_count: u16,
    media_present: u8,
    write_prot: u8,
    protection_type: u8,
    provisioning_mode: u8,
    zeroing_mode: u8,
    nr_actuators: u8,
    suspended: bool,
    __slate_bits_0: __slate_bits::__SlateBits854U0,
}


fn scsi_disk(disk: *mut gendisk) -> *mut scsi_disk {
(unsafe { (*disk).private_data }) as *mut scsi_disk
}

fn scsi_medium_access_command(scmd: *mut scsi_cmnd) -> i32 {
match (unsafe { *((*scmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32 {
8 | 10 | 40 | 42 | 47 | 53 | 65 | 66 | 136 | 138 | 143 | 147 | 168 | 170 | 175 => {
return 1;
}
127 => {
match (unsafe { *((*scmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(9 as isize) }) as i32 {
9..=11 | 13 => {
return 1;
}
_ => {
}
}
}
_ => {
}
}
0
}

fn logical_to_sectors(sdev: *mut scsi_device, blocks: u64) -> u64 {
blocks << ilog2!(unsafe { (*sdev).sector_size }).wrapping_sub(9)
}

fn logical_to_bytes(sdev: *mut scsi_device, blocks: u64) -> u32 {
blocks.wrapping_mul((unsafe { (*sdev).sector_size }) as u64) as u32
}

fn bytes_to_logical(sdev: *mut scsi_device, bytes: u32) -> u64 {
(bytes >> ilog2!(unsafe { (*sdev).sector_size })) as u64
}

fn sectors_to_logical(sdev: *mut scsi_device, sector: u64) -> u64 {
sector >> ilog2!(unsafe { (*sdev).sector_size }).wrapping_sub(9)
}

fn sd_zbc_read_zones(sdkp: *mut scsi_disk, lim: *mut queue_limits, buf: *mut u8) -> i32 {
0
}

fn sd_zbc_revalidate_zones(sdkp: *mut scsi_disk) -> i32 {
0
}

fn sd_zbc_setup_zone_mgmt_cmnd(cmd: *mut scsi_cmnd, op: u8, all: bool) -> u8 {
5
}

fn sd_zbc_complete(cmd: *mut scsi_cmnd, good_bytes: u32, sshdr: &scsi_sense_hdr) -> u32 {
good_bytes
}

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_593: [u8; 29] = [115, 100, 95, 109, 111, 100, 46, 97, 117, 116, 104, 111, 114, 61, 69, 114, 105, 99, 32, 89, 111, 117, 110, 103, 100, 97, 108, 101, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_594: [u8; 41] = [115, 100, 95, 109, 111, 100, 46, 100, 101, 115, 99, 114, 105, 112, 116, 105, 111, 110, 61, 83, 67, 83, 73, 32, 100, 105, 115, 107, 32, 40, 115, 100, 41, 32, 100, 114, 105, 118, 101, 114, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_595: [u8; 32] = [115, 100, 95, 109, 111, 100, 46, 102, 105, 108, 101, 61, 100, 114, 105, 118, 101, 114, 115, 47, 115, 99, 115, 105, 47, 115, 100, 95, 109, 111, 100, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_596: [u8; 19] = [115, 100, 95, 109, 111, 100, 46, 108, 105, 99, 101, 110, 115, 101, 61, 71, 80, 76, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_597: [u8; 29] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 56, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_598: [u8; 30] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 54, 53, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_599: [u8; 30] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 54, 54, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_600: [u8; 30] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 54, 55, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_601: [u8; 30] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 54, 56, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_602: [u8; 30] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 54, 57, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_603: [u8; 30] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 55, 48, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_604: [u8; 30] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 55, 49, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_605: [u8; 31] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 49, 50, 56, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_606: [u8; 31] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 49, 50, 57, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_607: [u8; 31] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 49, 51, 48, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_608: [u8; 31] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 49, 51, 49, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_609: [u8; 31] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 49, 51, 50, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_610: [u8; 31] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 49, 51, 51, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_611: [u8; 31] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 49, 51, 52, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_612: [u8; 31] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 98, 108, 111, 99, 107, 45, 109, 97, 106, 111, 114, 45, 49, 51, 53, 45, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_613: [u8; 26] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 115, 99, 115, 105, 58, 116, 45, 48, 120, 48, 48, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_614: [u8; 26] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 115, 99, 115, 105, 58, 116, 45, 48, 120, 48, 55, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_615: [u8; 26] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 115, 99, 115, 105, 58, 116, 45, 48, 120, 48, 101, 42, 0];

#[used(compiler)]
#[unsafe(link_section = ".modinfo")]
static __UNIQUE_ID_modinfo_616: [u8; 26] = [115, 100, 95, 109, 111, 100, 46, 97, 108, 105, 97, 115, 61, 115, 99, 115, 105, 58, 116, 45, 48, 120, 49, 52, 42, 0];

fn sd_config_write_same(sdkp: *mut scsi_disk, lim: *mut queue_limits) {
let mut __UNIQUE_ID_y__636: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__635: u32 = unsafe { core::mem::zeroed() };
let mut __v3: u32 = unsafe { core::mem::zeroed() };
let mut __v4: u32 = unsafe { core::mem::zeroed() };
let mut __y: u32 = unsafe { core::mem::zeroed() };
let mut __x: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__639: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__638: u32 = unsafe { core::mem::zeroed() };
let mut __v11: u32 = unsafe { core::mem::zeroed() };
let mut __v12: u32 = unsafe { core::mem::zeroed() };
let mut __y_1: u32 = unsafe { core::mem::zeroed() };
let mut __x_1: u32 = unsafe { core::mem::zeroed() };
let mut logical_block_size: u32 = unsafe { core::mem::zeroed() };
unsafe {
logical_block_size = (*(*sdkp).device).sector_size;
if ((*(*sdkp).device).__slate_bits_0.__get_no_write_same() as i32) != 0 {
(*sdkp).max_ws_blocks = 0;
} else {
// Some devices can not handle block counts above 0xffff despite
// supporting WRITE SAME(16). Consequently we default to 64k
// blocks per I/O unless the device explicitly advertises a
// bigger limit.
if (*sdkp).max_ws_blocks > 65535 {
__x = (*sdkp).max_ws_blocks;
__y = 8388607;
if __x == 0 {
__v4 = __y;
} else {
if __y == 0 {
__v3 = __x;
} else {
__UNIQUE_ID_x__635 = __x;
__UNIQUE_ID_y__636 = __y;
while false {
}
__v3 = if __UNIQUE_ID_x__635 < __UNIQUE_ID_y__636 { __UNIQUE_ID_x__635 } else { __UNIQUE_ID_y__636 };
}
__v4 = __v3;
}
(*sdkp).max_ws_blocks = __v4;
} else {
if ((*sdkp).__slate_bits_0.__get_ws16() as i32) != 0 || ((*sdkp).__slate_bits_0.__get_ws10() as i32) != 0 || ((*(*sdkp).device).__slate_bits_0.__get_no_report_opcodes() as i32) != 0 {
__x_1 = (*sdkp).max_ws_blocks;
__y_1 = 65535;
if __x_1 == 0 {
__v12 = __y_1;
} else {
if __y_1 == 0 {
__v11 = __x_1;
} else {
__UNIQUE_ID_x__638 = __x_1;
__UNIQUE_ID_y__639 = __y_1;
while false {
}
__v11 = if __UNIQUE_ID_x__638 < __UNIQUE_ID_y__639 { __UNIQUE_ID_x__638 } else { __UNIQUE_ID_y__639 };
}
__v12 = __v11;
}
(*sdkp).max_ws_blocks = __v12;
} else {
(*(*sdkp).device).__slate_bits_0.__set_no_write_same(1u32);
(*sdkp).max_ws_blocks = 0;
}
}
if ((*sdkp).__slate_bits_0.__get_lbprz() as i32) != 0 && ((*sdkp).__slate_bits_0.__get_lbpws() as i32) != 0 {
(*sdkp).zeroing_mode = 2;
} else {
if ((*sdkp).__slate_bits_0.__get_lbprz() as i32) != 0 && ((*sdkp).__slate_bits_0.__get_lbpws10() as i32) != 0 {
(*sdkp).zeroing_mode = 3;
} else {
if (*sdkp).max_ws_blocks != 0 {
(*sdkp).zeroing_mode = 1;
} else {
(*sdkp).zeroing_mode = 0;
}
}
}
if (*sdkp).max_ws_blocks != 0 && (*sdkp).physical_block_size > logical_block_size {
// Reporting a maximum number of blocks that is not aligned
// on the device physical size would cause a large write same
// request to be split into physically unaligned chunks by
// __blkdev_issue_write_zeroes() even if the caller of this
// functions took care to align the large request. So make sure
// the maximum reported is aligned to the device physical block
// size. This is only an optional optimization for regular
// disks, but this is mandatory to avoid failure of large write
// same requests directed at sequential write required zones of
// host-managed ZBC disks.
(*sdkp).max_ws_blocks = round_down!(unsafe { (*sdkp).max_ws_blocks }, bytes_to_logical((*sdkp).device, (*sdkp).physical_block_size));
}
}
(*lim).max_write_zeroes_sectors = unsafe { (*sdkp).max_ws_blocks }.wrapping_mul(logical_block_size >> 9i32);
if ((*sdkp).zeroing_mode as i32) == 2 || ((*sdkp).zeroing_mode as i32) == 3 {
(*lim).max_hw_wzeroes_unmap_sectors = (*lim).max_write_zeroes_sectors;
}
}
}

fn sd_revalidate_disk(disk: *mut gendisk) {
let mut __slate_compound_37: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_39: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_40: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_42: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
// Query concurrent positioning ranges after
// queue_limits_commit_update() unlocked q->limits_lock to avoid
// deadlock with q->sysfs_dir_lock and q->sysfs_lock.
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__672: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__671: u32 = unsafe { core::mem::zeroed() };
let mut __v4: u32 = unsafe { core::mem::zeroed() };
let mut __v5: u32 = unsafe { core::mem::zeroed() };
let mut __y: u32 = unsafe { core::mem::zeroed() };
let mut __x: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__669: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__668: u32 = unsafe { core::mem::zeroed() };
let mut __v12: u32 = unsafe { core::mem::zeroed() };
let mut __v13: u32 = unsafe { core::mem::zeroed() };
let mut __y_1: u32 = unsafe { core::mem::zeroed() };
let mut __x_1: u32 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_20: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_21: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_25: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_30: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut dev_max: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u32 = core::ptr::addr_of_mut!(dev_max) as *mut u32;
let mut buffer: *mut u8 = unsafe { core::mem::zeroed() };
let mut lim: *mut queue_limits = unsafe { core::mem::zeroed() };
let mut old_capacity: u64 = unsafe { core::mem::zeroed() };
let mut sdp: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut sdkp: *mut scsi_disk = unsafe { core::mem::zeroed() };
unsafe {
sdkp = scsi_disk(disk);
sdp = (*sdkp).device;
old_capacity = (*sdkp).capacity;
lim = core::ptr::null_mut::<queue_limits>();
buffer = core::ptr::null_mut::<u8>();
let __slate_auto_bytes_3 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_30 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
SCSI_LOG_HLQUEUE!();
// If the device is offline, don't try and read capacity or any
// of the other niceties.
if scsi_device_online(sdp) == 0 {
return;
} else {
__obj_size = size_mul(core::mem::size_of::<queue_limits>() as u64, 1u64);
let __slate_auto_bytes_2 = __slate_slot_25 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_24 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_25 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_40 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_40)
});
alloc_tag_restore!();
} else {
*__slate_slot_25 = _kmalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_42 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_42)
});
}
lim = *__slate_slot_25 as *mut queue_limits;
if lim.is_null() {
return;
} else {
let __slate_auto_bytes_1 = __slate_slot_21 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_20 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_20 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_21 = _kmalloc_noprof(512u64, 3264u32, *{
__slate_compound_37 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_37)
});
alloc_tag_restore!();
} else {
*__slate_slot_21 = _kmalloc_noprof(512u64, 3264u32, *{
__slate_compound_39 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_39)
});
}
buffer = *__slate_slot_21 as *mut u8;
if !buffer.is_null() {
sd_spinup_disk(sdkp);
*lim = queue_limits_start_update((*(*sdkp).disk).queue);
// Without media there is no reason to ask; moreover, some devices
// react badly if we do.
if (*sdkp).media_present != 0 {
sd_read_capacity(sdkp, lim, buffer);
// Some USB/UAS devices return generic values for mode pages
// until the media has been accessed. Trigger a READ operation
// to force the device to populate mode pages.
if ((*sdp).__slate_bits_0.__get_read_before_ms() as i32) != 0 {
sd_read_block_zero(sdkp);
}
// set the default to rotational.  All non-rotational devices
// support the block characteristics VPD page, which will
// cause this to be updated correctly and any device which
// doesn't support it should be treated as rotational.
(*lim).features = (*lim).features | 12u32;
if scsi_device_supports_vpd(sdp) != 0 {
sd_read_block_provisioning(sdkp);
sd_read_block_limits(sdkp, lim);
sd_read_block_limits_ext(sdkp);
sd_read_block_characteristics(sdkp, lim);
sd_zbc_read_zones(sdkp, lim, buffer);
}
sd_config_discard(sdkp, lim, sd_discard_mode(sdkp));
sd_print_capacity(sdkp, old_capacity);
sd_read_write_protect_flag(sdkp, buffer);
sd_read_cache_type(sdkp, buffer);
sd_read_io_hints(sdkp, buffer);
sd_read_app_tag_own(sdkp, buffer);
sd_read_write_same(sdkp, buffer);
sd_read_security(sdkp, buffer);
sd_config_protection(sdkp, lim);
}
// We now have all cache related info, determine how we deal
// with flush requests.
sd_set_flush_flag(sdkp, lim);
// Initial block count limit based on CDB TRANSFER LENGTH field size.
*__slate_slot_31 = if ((*sdp).__slate_bits_0.__get_use_16_for_rw() as i32) != 0 { 4294967295 } else { 65535 };
__x_1 = *__slate_slot_31;
__y_1 = (*sdkp).max_xfer_blocks;
if __x_1 == 0 {
__v13 = __y_1;
} else {
if __y_1 == 0 {
__v12 = __x_1;
} else {
__UNIQUE_ID_x__668 = __x_1;
__UNIQUE_ID_y__669 = __y_1;
while false {
}
__v12 = if __UNIQUE_ID_x__668 < __UNIQUE_ID_y__669 { __UNIQUE_ID_x__668 } else { __UNIQUE_ID_y__669 };
}
__v13 = __v12;
}
// Some devices report a maximum block count for READ/WRITE requests.
*__slate_slot_31 = __v13;
(*lim).max_dev_sectors = logical_to_sectors(sdp, *__slate_slot_31 as u64) as u32;
if sd_validate_min_xfer_size(sdkp) {
(*lim).io_min = logical_to_bytes(sdp, (*sdkp).min_xfer_blocks as u64);
} else {
(*lim).io_min = 0;
}
// Limit default to SCSI host optimal sector limit if set. There may be
// an impact on performance for when the size of a request exceeds this
// host limit.
(*lim).io_opt = (*(*sdp).host).opt_sectors << 9i32;
if sd_validate_opt_xfer_size(sdkp, *__slate_slot_31) {
__x = (*lim).io_opt;
__y = logical_to_bytes(sdp, (*sdkp).opt_xfer_blocks as u64);
if __x == 0 {
__v5 = __y;
} else {
if __y == 0 {
__v4 = __x;
} else {
__UNIQUE_ID_x__671 = __x;
__UNIQUE_ID_y__672 = __y;
while false {
}
__v4 = if __UNIQUE_ID_x__671 < __UNIQUE_ID_y__672 { __UNIQUE_ID_x__671 } else { __UNIQUE_ID_y__672 };
}
__v5 = __v4;
}
(*lim).io_opt = __v5;
}
(*sdkp).__slate_bits_0.__set_first_scan(0u32);
set_capacity_and_notify(disk, logical_to_sectors(sdp, (*sdkp).capacity));
sd_config_write_same(sdkp, lim);
*__slate_slot_30 = queue_limits_commit_update_frozen((*(*sdkp).disk).queue, lim);
if *__slate_slot_30 == 0 {
if (*sdkp).media_present != 0 {
__v0 = scsi_device_supports_vpd(sdp) != 0;
} else {
__v0 = false;
}
if __v0 {
sd_read_cpr(sdkp);
}
// For a zoned drive, revalidating the zones can be done only once
// the gendisk capacity is set. So if this fails, set back the gendisk
// capacity to 0.
if sd_zbc_revalidate_zones(sdkp) != 0 {
set_capacity_and_notify(disk, 0u64);
}
}
}
kfree(buffer as *const ());
kfree(lim as *const ());
}
}
}
}

static mut sd_index_ida: ida = ida { xa: xarray { xa_lock: spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} }, xa_flags: 16777221u32, xa_head: core::ptr::null_mut::<()>() } };

static mut sd_mutex_lock: mutex = {
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

static mut sd_page_pool: *mut mempool = unsafe { core::mem::zeroed() };

static mut sd_large_page_pool: *mut mempool = unsafe { core::mem::zeroed() };

static mut sd_large_page_pool_users: __SlateRecord_types_h_190 = __SlateRecord_types_h_190 { counter: 0i32 };

static mut sd_bio_compl_lkclass: lock_class_key = unsafe { core::mem::zeroed() };

static mut sd_cache_types: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"write through\0".as_ptr() as *mut u8) as *const u8, (b"none\0".as_ptr() as *mut u8) as *const u8, (b"write back\0".as_ptr() as *mut u8) as *const u8, (b"write back, no read (daft)\0".as_ptr() as *mut u8) as *const u8]);

fn sd_large_pool_create() -> i32 {
unsafe { mutex_lock(core::ptr::addr_of_mut!(sd_mutex_lock)) };
if unsafe { sd_large_page_pool }.is_null() {
unsafe {
sd_large_page_pool = mempool_create_page_pool!(2i32, get_order(4096u64));
}
if unsafe { sd_large_page_pool }.is_null() {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013sd: can't create large page mempool\n".as_ptr() as *const u8) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!(sd_mutex_lock)) };
return 12i32.wrapping_neg();
}
}
atomic_inc(unsafe { core::ptr::addr_of_mut!(sd_large_page_pool_users) });
unsafe { mutex_unlock(core::ptr::addr_of_mut!(sd_mutex_lock)) };
0
}

fn sd_large_pool_destroy() {
unsafe { mutex_lock(core::ptr::addr_of_mut!(sd_mutex_lock)) };
if atomic_dec_and_test(unsafe { core::ptr::addr_of_mut!(sd_large_page_pool_users) }) {
unsafe { mempool_destroy(sd_large_page_pool) };
unsafe {
sd_large_page_pool = core::ptr::null_mut::<mempool>();
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(sd_mutex_lock)) };
}

fn sd_disable_discard(sdkp: *mut scsi_disk) {
unsafe {
(*sdkp).provisioning_mode = 5;
}
blk_queue_disable_discard(unsafe { (*(*sdkp).disk).queue });
}

fn sd_config_discard(sdkp: *mut scsi_disk, lim: *mut queue_limits, mode: u32) {
let logical_block_size: u32 = unsafe { (*(*sdkp).device).sector_size };
let mut max_blocks: u32 = 0;
unsafe {
(*lim).discard_alignment = unsafe { (*sdkp).unmap_alignment }.wrapping_mul(logical_block_size);
}
let mut __v0: u32;
let __UNIQUE_ID_x__617: u32 = unsafe { (*sdkp).physical_block_size };
let __UNIQUE_ID_y__618: u32 = unsafe { (*sdkp).unmap_granularity }.wrapping_mul(logical_block_size);
loop {
if __types_ok!(__UNIQUE_ID_x__617, __UNIQUE_ID_y__618) == 0 {
unsafe { __compiletime_assert_619() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__617 > __UNIQUE_ID_y__618 { __UNIQUE_ID_x__617 } else { __UNIQUE_ID_y__618 };
unsafe {
(*lim).discard_granularity = __v0;
}
unsafe {
(*sdkp).provisioning_mode = mode as u8;
}
match mode {
0 | 5 => {
}
1 => {
let mut __v3: u32;
let __x: u32 = unsafe { (*sdkp).max_unmap_blocks };
let __y: u32 = 8388607;
let __v4: u32 = if __x == 0 {
__y
} else {
let __v5: u32 = if __y == 0 {
__x
} else {
let mut __v6: u32;
let __UNIQUE_ID_x__620: u32 = __x;
let __UNIQUE_ID_y__621: u32 = __y;
loop {
if __types_ok!(__UNIQUE_ID_x__620, __UNIQUE_ID_y__621) == 0 {
unsafe { __compiletime_assert_622() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = if __UNIQUE_ID_x__620 < __UNIQUE_ID_y__621 { __UNIQUE_ID_x__620 } else { __UNIQUE_ID_y__621 };
__v6
};
__v5
};
__v3 = __v4;
max_blocks = __v3;
}
2 => {
if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_unmap_limit_for_ws() }) as i32) != 0 {
max_blocks = unsafe { (*sdkp).max_unmap_blocks };
} else {
max_blocks = unsafe { (*sdkp).max_ws_blocks };
}
let mut __v8: u32;
let __x: u32 = max_blocks;
let __y: u32 = 8388607;
let __v9: u32 = if __x == 0 {
__y
} else {
let __v10: u32 = if __y == 0 {
__x
} else {
let mut __v11: u32;
let __UNIQUE_ID_x__623: u32 = __x;
let __UNIQUE_ID_y__624: u32 = __y;
loop {
if __types_ok!(__UNIQUE_ID_x__623, __UNIQUE_ID_y__624) == 0 {
unsafe { __compiletime_assert_625() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v11 = if __UNIQUE_ID_x__623 < __UNIQUE_ID_y__624 { __UNIQUE_ID_x__623 } else { __UNIQUE_ID_y__624 };
__v11
};
__v10
};
__v8 = __v9;
max_blocks = __v8;
}
3 => {
if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_unmap_limit_for_ws() }) as i32) != 0 {
max_blocks = unsafe { (*sdkp).max_unmap_blocks };
} else {
max_blocks = unsafe { (*sdkp).max_ws_blocks };
}
let mut __v13: u32;
let __x: u32 = max_blocks;
let __y: u32 = 65535;
let __v14: u32 = if __x == 0 {
__y
} else {
let __v15: u32 = if __y == 0 {
__x
} else {
let mut __v16: u32;
let __UNIQUE_ID_x__626: u32 = __x;
let __UNIQUE_ID_y__627: u32 = __y;
loop {
if __types_ok!(__UNIQUE_ID_x__626, __UNIQUE_ID_y__627) == 0 {
unsafe { __compiletime_assert_628() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v16 = if __UNIQUE_ID_x__626 < __UNIQUE_ID_y__627 { __UNIQUE_ID_x__626 } else { __UNIQUE_ID_y__627 };
__v16
};
__v15
};
__v13 = __v14;
max_blocks = __v13;
}
4 => {
let mut __v18: u32;
let __x: u32 = unsafe { (*sdkp).max_ws_blocks };
let __y: u32 = 65535;
let __v19: u32 = if __x == 0 {
__y
} else {
let __v20: u32 = if __y == 0 {
__x
} else {
let mut __v21: u32;
let __UNIQUE_ID_x__629: u32 = __x;
let __UNIQUE_ID_y__630: u32 = __y;
loop {
if __types_ok!(__UNIQUE_ID_x__629, __UNIQUE_ID_y__630) == 0 {
unsafe { __compiletime_assert_631() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v21 = if __UNIQUE_ID_x__629 < __UNIQUE_ID_y__630 { __UNIQUE_ID_x__629 } else { __UNIQUE_ID_y__630 };
__v21
};
__v20
};
__v18 = __v19;
max_blocks = __v18;
}
_ => {
}
}
unsafe {
(*lim).max_hw_discard_sectors = max_blocks.wrapping_mul(logical_block_size >> 9i32);
}
}

fn sd_set_flush_flag(sdkp: *mut scsi_disk, lim: *mut queue_limits) {
if ((unsafe { (*sdkp).__slate_bits_0.__get_WCE() }) as i32) != 0 {
unsafe {
(*lim).features = (*lim).features | 1u32;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_DPOFUA() }) as i32) != 0 {
unsafe {
(*lim).features = (*lim).features | 2u32;
}
} else {
let __v8: u32 = (unsafe { (*lim).features }) & 4294967293u32;
unsafe {
(*lim).features = __v8;
}
}
} else {
let __v11: u32 = (unsafe { (*lim).features }) & 4294967292u32;
unsafe {
(*lim).features = __v11;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.sd.cache_type_store")]
extern "C" fn cache_type_store(dev: *mut device, attr: *mut device_attribute, mut buf: *const u8, count: u64) -> i64 {
let mut ct: i32 = 0;
let mut rcd: i32 = 0;
let mut wce: i32 = 0;
let mut sp: i32 = 0;
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut buffer: __SlateAlign16<[u8; 64]> = __SlateAlign16([0u8; 64]);
let mut data: scsi_mode_data = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(data) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut len: i32 = 0;
let mut ret: i32 = 0;
if ((unsafe { (*sdp).r#type }) as i32) != 0 && ((unsafe { (*sdp).r#type }) as i32) != 20 {
return 22i32.wrapping_neg() as i64;
}
// no cache control on RBC devices; theoretically they
// can do it, but there's probably so many exceptions
// it's not worth the risk
if (unsafe { strncmp(buf, core::ptr::addr_of!(temp) as *const u8, (core::mem::size_of::<[u8; 11]>() as u64).wrapping_sub(1)) }) == 0 {
let __v1: *const u8 = buf;
buf = __v1.wrapping_offset((core::mem::size_of::<[u8; 11]>() as u64).wrapping_sub(1) as isize);
unsafe {
(*sdkp).__slate_bits_0.__set_cache_override(1u32);
}
} else {
unsafe {
(*sdkp).__slate_bits_0.__set_cache_override(0u32);
}
}
ct = sysfs_match_string!(unsafe { core::ptr::addr_of_mut!(sd_cache_types.0) as *mut *const u8 }, buf);
if ct < 0 {
return 22i32.wrapping_neg() as i64;
}
rcd = if ct & 1 != 0 { 1 } else { 0 };
wce = if ct & 2 != 0 && (unsafe { (*sdkp).write_prot }) == 0 { 1 } else { 0 };
if ((unsafe { (*sdkp).__slate_bits_0.__get_cache_override() }) as i32) != 0 {
let mut lim: queue_limits = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(lim) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 192usize);
}
unsafe {
(*sdkp).__slate_bits_0.__set_WCE(wce as u32);
}
unsafe {
(*sdkp).__slate_bits_0.__set_RCD(rcd as u32);
}
lim = queue_limits_start_update(unsafe { (*(*sdkp).disk).queue });
sd_set_flush_flag(sdkp, core::ptr::addr_of_mut!(lim));
ret = unsafe { queue_limits_commit_update_frozen((*(*sdkp).disk).queue, core::ptr::addr_of_mut!(lim)) };
if ret != 0 {
return ret as i64;
}
return count as i64;
}
if (unsafe { scsi_mode_sense(sdp, 8i32, 8i32, 0i32, buffer.0.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 64]>() as i32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of_mut!(data), core::ptr::null_mut::<scsi_sense_hdr>()) }) != 0 {
return 22i32.wrapping_neg() as i64;
}
len = min_t__2!(u64, core::mem::size_of::<[u8; 64]>() as u64, data.length.wrapping_sub((data.header_length as i32) as u32).wrapping_sub((data.block_descriptor_length as i32) as u32)) as i32;
let buffer_data: *mut u8 = (buffer.0.as_mut_ptr() as *mut u8).wrapping_offset((data.header_length as i32) as isize).wrapping_offset((data.block_descriptor_length as i32) as isize);
let __v6: *mut u8 = buffer_data.wrapping_offset(2 as isize);
let __v8: u8 = (((unsafe { *__v6 }) as i32) & !5) as u8;
unsafe {
*__v6 = __v8;
}
let __v9: *mut u8 = buffer_data.wrapping_offset(2 as isize);
unsafe {
*__v9 = ((*__v9 as i32) | (wce << 2i32 | rcd)) as u8;
}
sp = if ((unsafe { *buffer_data.wrapping_offset(0 as isize) }) as i32) & 128 != 0 { 1 } else { 0 };
let __v12: *mut u8 = buffer_data.wrapping_offset(0 as isize);
let __v14: u8 = (((unsafe { *__v12 }) as i32) & !128) as u8;
unsafe {
*__v12 = __v14;
}
// Ensure WP, DPOFUA, and RESERVED fields are cleared in
// received mode parameter buffer before doing MODE SELECT.
data.device_specific = 0;
ret = unsafe { scsi_mode_select(sdp, 1i32, sp, buffer_data as *mut u8, len, 30000i32, (*sdkp).max_retries, core::ptr::addr_of_mut!(data), core::ptr::addr_of_mut!(sshdr)) };
if ret != 0 {
let __v15: bool = ret > 0 && scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr);
if __v15 {
sd_print_sense_hdr(sdkp, core::ptr::addr_of_mut!(sshdr));
}
return 22i32.wrapping_neg() as i64;
}
sd_revalidate_disk(unsafe { (*sdkp).disk });
count as i64
}

static temp: [u8; 11] = [116, 101, 109, 112, 111, 114, 97, 114, 121, 32, 0];

#[unsafe(link_section = ".text.slate_distinct.sd.manage_start_stop_show")]
extern "C" fn manage_start_stop_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
(unsafe { sysfs_emit(buf, c"%u\n".as_ptr() as *const u8, (((*sdp).__slate_bits_0.__get_manage_system_start_stop() as i32) != 0 && ((*sdp).__slate_bits_0.__get_manage_runtime_start_stop() as i32) != 0 && ((*sdp).__slate_bits_0.__get_manage_shutdown() as i32) != 0) as i32) }) as i64
}

static mut dev_attr_manage_start_stop: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"manage_start_stop\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_309>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_309_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_309_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_309_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_309_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_309_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(manage_start_stop_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

#[unsafe(link_section = ".text.slate_distinct.sd.manage_system_start_stop_show")]
extern "C" fn manage_system_start_stop_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
(unsafe { sysfs_emit(buf, c"%u\n".as_ptr() as *const u8, (*sdp).__slate_bits_0.__get_manage_system_start_stop() as i32) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.manage_system_start_stop_store")]
extern "C" fn manage_system_start_stop_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut v: bool = false;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
if (unsafe { kstrtobool(buf, core::ptr::addr_of_mut!(v)) }) != 0 {
return 22i32.wrapping_neg() as i64;
}
unsafe {
(*sdp).__slate_bits_0.__set_manage_system_start_stop(v as u32);
}
count as i64
}

static mut dev_attr_manage_system_start_stop: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"manage_system_start_stop\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_340>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_340_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_340_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_340_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_340_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_340_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(manage_system_start_stop_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(manage_system_start_stop_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.manage_runtime_start_stop_show")]
extern "C" fn manage_runtime_start_stop_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
(unsafe { sysfs_emit(buf, c"%u\n".as_ptr() as *const u8, (*sdp).__slate_bits_0.__get_manage_runtime_start_stop() as i32) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.manage_runtime_start_stop_store")]
extern "C" fn manage_runtime_start_stop_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut v: bool = false;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
if (unsafe { kstrtobool(buf, core::ptr::addr_of_mut!(v)) }) != 0 {
return 22i32.wrapping_neg() as i64;
}
unsafe {
(*sdp).__slate_bits_0.__set_manage_runtime_start_stop(v as u32);
}
count as i64
}

static mut dev_attr_manage_runtime_start_stop: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"manage_runtime_start_stop\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_371>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_371_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_371_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_371_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_371_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_371_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(manage_runtime_start_stop_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(manage_runtime_start_stop_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.manage_shutdown_show")]
extern "C" fn manage_shutdown_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
(unsafe { sysfs_emit(buf, c"%u\n".as_ptr() as *const u8, (*sdp).__slate_bits_0.__get_manage_shutdown() as i32) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.manage_shutdown_store")]
extern "C" fn manage_shutdown_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut v: bool = false;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
if (unsafe { kstrtobool(buf, core::ptr::addr_of_mut!(v)) }) != 0 {
return 22i32.wrapping_neg() as i64;
}
unsafe {
(*sdp).__slate_bits_0.__set_manage_shutdown(v as u32);
}
count as i64
}

static mut dev_attr_manage_shutdown: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"manage_shutdown\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_400>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_400_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_400_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_400_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_400_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_400_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(manage_shutdown_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(manage_shutdown_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.manage_restart_show")]
extern "C" fn manage_restart_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
(unsafe { sysfs_emit(buf, c"%u\n".as_ptr() as *const u8, (*sdp).__slate_bits_0.__get_manage_restart() as i32) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.manage_restart_store")]
extern "C" fn manage_restart_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut v: bool = false;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
if (unsafe { kstrtobool(buf, core::ptr::addr_of_mut!(v)) }) != 0 {
return 22i32.wrapping_neg() as i64;
}
unsafe {
(*sdp).__slate_bits_0.__set_manage_restart(v as u32);
}
count as i64
}

static mut dev_attr_manage_restart: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"manage_restart\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_429>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_429_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_429_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_429_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_429_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_429_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(manage_restart_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(manage_restart_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.allow_restart_show")]
extern "C" fn allow_restart_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%u\n".as_ptr() as *const u8, (*(*sdkp).device).__slate_bits_0.__get_allow_restart() as i32) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.allow_restart_store")]
extern "C" fn allow_restart_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let mut v: bool = false;
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
if ((unsafe { (*sdp).r#type }) as i32) != 0 && ((unsafe { (*sdp).r#type }) as i32) != 20 {
return 22i32.wrapping_neg() as i64;
}
if (unsafe { kstrtobool(buf, core::ptr::addr_of_mut!(v)) }) != 0 {
return 22i32.wrapping_neg() as i64;
}
unsafe {
(*sdp).__slate_bits_0.__set_allow_restart(v as u32);
}
count as i64
}

static mut dev_attr_allow_restart: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"allow_restart\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_460>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_460_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_460_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_460_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_460_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_460_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(allow_restart_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(allow_restart_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.cache_type_show")]
extern "C" fn cache_type_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let ct: i32 = ((unsafe { (*sdkp).__slate_bits_0.__get_RCD() }) as i32).wrapping_add(2i32.wrapping_mul((unsafe { (*sdkp).__slate_bits_0.__get_WCE() }) as i32));
(unsafe { sprintf(buf, c"%s\n".as_ptr() as *const u8, sd_cache_types.0[ct as usize]) }) as i64
}

static mut dev_attr_cache_type: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"cache_type\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_470>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_470_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_470_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_470_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_470_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_470_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(cache_type_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(cache_type_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.FUA_show")]
extern "C" fn FUA_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%u\n".as_ptr() as *const u8, (*sdkp).__slate_bits_0.__get_DPOFUA() as i32) }) as i64
}

static mut dev_attr_FUA: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"FUA\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_479>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_479_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_479_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_479_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_479_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_479_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(FUA_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

#[unsafe(link_section = ".text.slate_distinct.sd.protection_type_show")]
extern "C" fn protection_type_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%u\n".as_ptr() as *const u8, (*sdkp).protection_type as i32) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.protection_type_store")]
extern "C" fn protection_type_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let mut val: u32 = 0;
let mut err: i32 = 0;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
err = unsafe { kstrtouint(buf, 10u32, core::ptr::addr_of_mut!(val)) };
if err != 0 {
return err as i64;
}
if val <= 3 {
unsafe {
(*sdkp).protection_type = val as u8;
}
}
count as i64
}

static mut dev_attr_protection_type: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"protection_type\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_511>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_511_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_511_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_511_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_511_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_511_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(protection_type_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(protection_type_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.protection_mode_show")]
extern "C" fn protection_mode_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut dif: u32 = scsi_host_dif_capable(unsafe { (*sdp).host }, (unsafe { (*sdkp).protection_type }) as u32);
let mut dix: u32 = scsi_host_dix_capable(unsafe { (*sdp).host }, (unsafe { (*sdkp).protection_type }) as u32);
let __v1: bool = dix == 0 && scsi_host_dix_capable(unsafe { (*sdp).host }, 0u32) != 0;
if __v1 {
dif = 0;
dix = 1;
}
if dif == 0 && dix == 0 {
return (unsafe { sprintf(buf, c"none\n".as_ptr() as *const u8) }) as i64;
}
(unsafe { sprintf(buf, c"%s%u\n".as_ptr() as *const u8, if dix != 0 { c"dix".as_ptr() as *mut u8 } else { c"dif".as_ptr() as *mut u8 }, dif) }) as i64
}

static mut dev_attr_protection_mode: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"protection_mode\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_534>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_534_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_534_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_534_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_534_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_534_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(protection_mode_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

#[unsafe(link_section = ".text.slate_distinct.sd.app_tag_own_show")]
extern "C" fn app_tag_own_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%u\n".as_ptr() as *const u8, (*sdkp).__slate_bits_0.__get_ATO() as i32) }) as i64
}

static mut dev_attr_app_tag_own: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"app_tag_own\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_543>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_543_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_543_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_543_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_543_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_543_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(app_tag_own_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

#[unsafe(link_section = ".text.slate_distinct.sd.thin_provisioning_show")]
extern "C" fn thin_provisioning_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%u\n".as_ptr() as *const u8, (*sdkp).__slate_bits_0.__get_lbpme() as i32) }) as i64
}

static mut dev_attr_thin_provisioning: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"thin_provisioning\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_553>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_553_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_553_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_553_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_553_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_553_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(thin_provisioning_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

// sysfs_match_string() requires dense arrays
static mut lbp_mode: __SlateAlign16<[*const u8; 6]> = __SlateAlign16([(b"full\0".as_ptr() as *mut u8) as *const u8, (b"unmap\0".as_ptr() as *mut u8) as *const u8, (b"writesame_16\0".as_ptr() as *mut u8) as *const u8, (b"writesame_10\0".as_ptr() as *mut u8) as *const u8, (b"writesame_zero\0".as_ptr() as *mut u8) as *const u8, (b"disabled\0".as_ptr() as *mut u8) as *const u8]);

#[unsafe(link_section = ".text.slate_distinct.sd.provisioning_mode_show")]
extern "C" fn provisioning_mode_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%s\n".as_ptr() as *const u8, lbp_mode.0[((*sdkp).provisioning_mode as i32) as usize]) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.provisioning_mode_store")]
extern "C" fn provisioning_mode_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut lim: queue_limits = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(lim) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 192usize);
}
let mut mode: i32 = 0;
let mut err: i32 = 0;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
if ((unsafe { (*sdp).r#type }) as i32) != 0 {
return 22i32.wrapping_neg() as i64;
}
mode = sysfs_match_string__2!(unsafe { core::ptr::addr_of_mut!(lbp_mode.0) as *mut *const u8 }, buf);
if mode < 0 {
return 22i32.wrapping_neg() as i64;
}
lim = queue_limits_start_update(unsafe { (*(*sdkp).disk).queue });
sd_config_discard(sdkp, core::ptr::addr_of_mut!(lim), mode as u32);
err = unsafe { queue_limits_commit_update_frozen((*(*sdkp).disk).queue, core::ptr::addr_of_mut!(lim)) };
if err != 0 {
return err as i64;
}
count as i64
}

static mut dev_attr_provisioning_mode: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"provisioning_mode\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_600>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_600_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_600_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_600_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_600_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_600_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(provisioning_mode_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(provisioning_mode_store) } };

// sysfs_match_string() requires dense arrays
static mut zeroing_mode: __SlateAlign16<[*const u8; 4]> = __SlateAlign16([(b"write\0".as_ptr() as *mut u8) as *const u8, (b"writesame\0".as_ptr() as *mut u8) as *const u8, (b"writesame_16_unmap\0".as_ptr() as *mut u8) as *const u8, (b"writesame_10_unmap\0".as_ptr() as *mut u8) as *const u8]);

#[unsafe(link_section = ".text.slate_distinct.sd.zeroing_mode_show")]
extern "C" fn zeroing_mode_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%s\n".as_ptr() as *const u8, zeroing_mode.0[((*sdkp).zeroing_mode as i32) as usize]) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.zeroing_mode_store")]
extern "C" fn zeroing_mode_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let mut mode: i32 = 0;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
mode = sysfs_match_string__3!(unsafe { core::ptr::addr_of_mut!(zeroing_mode.0) as *mut *const u8 }, buf);
if mode < 0 {
return 22i32.wrapping_neg() as i64;
}
unsafe {
(*sdkp).zeroing_mode = mode as u8;
}
count as i64
}

static mut dev_attr_zeroing_mode: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"zeroing_mode\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_637>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_637_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_637_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_637_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_637_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_637_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(zeroing_mode_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(zeroing_mode_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.max_medium_access_timeouts_show")]
extern "C" fn max_medium_access_timeouts_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%u\n".as_ptr() as *const u8, (*sdkp).max_medium_access_timeouts) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.max_medium_access_timeouts_store")]
extern "C" fn max_medium_access_timeouts_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let mut err: i32 = 0;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
err = unsafe { kstrtouint(buf, 10u32, core::ptr::addr_of_mut!((*sdkp).max_medium_access_timeouts)) };
(if err != 0 { err as u64 } else { count }) as i64
}

static mut dev_attr_max_medium_access_timeouts: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"max_medium_access_timeouts\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_663>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_663_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_663_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_663_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_663_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_663_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(max_medium_access_timeouts_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(max_medium_access_timeouts_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.max_write_same_blocks_show")]
extern "C" fn max_write_same_blocks_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%u\n".as_ptr() as *const u8, (*sdkp).max_ws_blocks) }) as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.max_write_same_blocks_store")]
extern "C" fn max_write_same_blocks_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut lim: queue_limits = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(lim) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 192usize);
}
let mut max: u64 = 0;
let mut err: i32 = 0;
if !unsafe { capable(21i32) } {
return 13i32.wrapping_neg() as i64;
}
if ((unsafe { (*sdp).r#type }) as i32) != 0 && ((unsafe { (*sdp).r#type }) as i32) != 20 {
return 22i32.wrapping_neg() as i64;
}
err = kstrtoul(buf, 10u32, core::ptr::addr_of_mut!(max));
if err != 0 {
return err as i64;
}
if max == 0 {
unsafe {
(*sdp).__slate_bits_0.__set_no_write_same(1u32);
}
} else {
if max <= 8388607 {
unsafe {
(*sdp).__slate_bits_0.__set_no_write_same(0u32);
}
unsafe {
(*sdkp).max_ws_blocks = max as u32;
}
}
}
lim = queue_limits_start_update(unsafe { (*(*sdkp).disk).queue });
sd_config_write_same(sdkp, core::ptr::addr_of_mut!(lim));
err = unsafe { queue_limits_commit_update_frozen((*(*sdkp).disk).queue, core::ptr::addr_of_mut!(lim)) };
if err != 0 {
return err as i64;
}
count as i64
}

static mut dev_attr_max_write_same_blocks: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"max_write_same_blocks\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_709>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_709_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_709_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_709_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_709_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_709_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(max_write_same_blocks_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(max_write_same_blocks_store) } };

#[unsafe(link_section = ".text.slate_distinct.sd.zoned_cap_show")]
extern "C" fn zoned_cap_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
if ((unsafe { (*(*sdkp).device).r#type }) as i32) == 20 {
return (unsafe { sprintf(buf, c"host-managed\n".as_ptr() as *const u8) }) as i64;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_zoned() }) as i32) == 1 {
return (unsafe { sprintf(buf, c"host-aware\n".as_ptr() as *const u8) }) as i64;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_zoned() }) as i32) == 2 {
return (unsafe { sprintf(buf, c"drive-managed\n".as_ptr() as *const u8) }) as i64;
}
(unsafe { sprintf(buf, c"none\n".as_ptr() as *const u8) }) as i64
}

static mut dev_attr_zoned_cap: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"zoned_cap\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_724>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_724_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_724_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_724_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_724_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_724_6>() as u64) as i32).wrapping_add(292i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(zoned_cap_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: None } };

#[unsafe(link_section = ".text.slate_distinct.sd.max_retries_store")]
extern "C" fn max_retries_store(dev: *mut device, attr: *mut device_attribute, buf: *const u8, count: u64) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
let mut retries: i32 = 0;
let err: i32 = unsafe { kstrtoint(buf, 10u32, core::ptr::addr_of_mut!(retries)) };
if err != 0 {
return err as i64;
}
if retries == 1i32.wrapping_neg() || retries <= 5 {
unsafe {
(*sdkp).max_retries = retries;
}
return count as i64;
}
unsafe { sdev_prefix_printk(c"\x013".as_ptr() as *const u8, sdev as *const scsi_device, core::ptr::null::<u8>(), c"max_retries must be between -1 and %d\n".as_ptr() as *const u8, 5i32) };
22i32.wrapping_neg() as i64
}

#[unsafe(link_section = ".text.slate_distinct.sd.max_retries_show")]
extern "C" fn max_retries_show(dev: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
(unsafe { sprintf(buf, c"%d\n".as_ptr() as *const u8, (*sdkp).max_retries) }) as i64
}

static mut dev_attr_max_retries: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"max_retries\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_sd_c_757>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_757_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_757_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_757_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_757_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_sd_c_757_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(max_retries_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(max_retries_store) } };

static mut sd_disk_attrs: __SlateAlign16<[*mut attribute; 19]> = __SlateAlign16([unsafe { core::ptr::addr_of_mut!(dev_attr_cache_type.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_FUA.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_allow_restart.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_manage_start_stop.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_manage_system_start_stop.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_manage_runtime_start_stop.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_manage_shutdown.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_manage_restart.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_protection_type.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_protection_mode.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_app_tag_own.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_thin_provisioning.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_provisioning_mode.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_zeroing_mode.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_max_write_same_blocks.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_max_medium_access_timeouts.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_zoned_cap.attr) }, unsafe { core::ptr::addr_of_mut!(dev_attr_max_retries.attr) }, core::ptr::null_mut::<attribute>()]);

static sd_disk_group: __SlateConst<attribute_group> = __SlateConst({
let mut __t0: attribute_group = unsafe { core::mem::zeroed() };
__t0.__slate_anon_4 = __SlateRecord_sysfs_h_118 { attrs: unsafe { core::ptr::addr_of_mut!(sd_disk_attrs.0) as *mut *mut attribute } };
__t0
});

static mut sd_disk_groups: __SlateAlign16<[*const attribute_group; 2]> = __SlateAlign16([unsafe { core::ptr::addr_of!(sd_disk_group.0) }, core::ptr::null::<attribute_group>()]);

#[unsafe(link_section = ".text.slate_distinct.sd.scsi_disk_release")]
extern "C" fn scsi_disk_release(dev: *mut device) {
let sdkp: *mut scsi_disk = to_scsi_disk!(dev);
unsafe { ida_free(core::ptr::addr_of_mut!(sd_index_ida), (*sdkp).index) };
unsafe { put_device(core::ptr::addr_of_mut!((*(*sdkp).device).sdev_gendev)) };
free_opal_dev(unsafe { (*sdkp).opal_dev });
unsafe { kfree(sdkp as *const ()) };
}

static mut sd_disk_class: class = {
let mut __t0: class = unsafe { core::mem::zeroed() };
__t0.name = (b"scsi_disk\0".as_ptr() as *mut u8) as *const u8;
__t0.dev_groups = (unsafe { core::ptr::addr_of_mut!(sd_disk_groups.0) as *mut *const attribute_group }) as *const *const attribute_group;
__t0.dev_release = Some(scsi_disk_release);
__t0
};

/// Don't request a new module, as that could deadlock in multipath
/// environment.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_default_probe")]
extern "C" fn sd_default_probe(devt: u32) {
}

/// Device no to disk mapping:
///
///       major         disc2     disc  p1
///   |............|.............|....|....| <- dev_t
///    31        20 19          8 7  4 3  0
///
/// Inside a major, we have 16k disks, however mapped non-
/// contiguously. The first 16 disks are for major0, the next
/// ones with major1, ... Disk 256 is for major0 again, disk 272
/// for major1, ...
/// As we stay compatible with our numbering scheme, we can reuse
/// the well-know SCSI majors 8, 65--71, 136--143.
fn sd_major(major_idx: i32) -> i32 {
match major_idx {
0 => {
return 8;
}
1..=7 => {
return 65i32.wrapping_add(major_idx).wrapping_sub(1);
}
8..=15 => {
return 128i32.wrapping_add(major_idx).wrapping_sub(8);
}
_ => {
loop {
unsafe {
core::arch::asm!("634: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 634b - ., 3; .popsection\n# {0}", const 634 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_sd_major_0, sym __slate_asm_string_sd_major_1, const 831 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
return 0; // shut up gcc
}
}
unsafe { core::mem::zeroed() }
}

/// Look up the DIX operation based on whether the command is read or
/// write and whether dix and dif are enabled.
fn sd_prot_op(write: bool, dix: bool, dif: bool) -> u32 {
// Lookup table: bit 2 (write), bit 1 (dix), bit 0 (dif)
// wrt dix dif
//  0	0   0
//  0	0   1
//  0	1   0
//  0	1   1
//  1	0   0
//  1	0   1
//  1	1   0
//  1	1   1
unsafe { ops.0[((write as i32) << 2i32 | (dix as i32) << 1i32 | (dif as i32)) as usize] }
}

static ops: __SlateAlign16<[u32; 8]> = __SlateAlign16([0u32, 3u32, 1u32, 5u32, 0u32, 4u32, 2u32, 6u32]);

/// Returns a mask of the protection flags that are valid for a given DIX
/// operation.
fn sd_prot_flag_mask(prot_op: u32) -> u32 {
unsafe { flag_mask.0[prot_op as usize] }
}

static flag_mask: __SlateAlign16<[u32; 7]> = __SlateAlign16([0u32, 24u32, 30u32, 15u32, 9u32, 31u32, 31u32]);

fn sd_setup_protect_cmnd(scmd: *mut scsi_cmnd, dix: u32, dif: u32) -> u8 {
let rq: *mut request = scsi_cmd_to_rq(scmd);
let bio: *mut bio = unsafe { (*rq).bio };
let prot_op: u32 = sd_prot_op(rq_data_dir!(rq) != 0, dix != 0, dif != 0);
let mut protect: u32 = 0;
if dix != 0 {
// DIX Type 0, 1, 2, 3
if bio_integrity_flagged(bio, 8u32) {
unsafe {
(*scmd).prot_flags = (((*scmd).prot_flags as i32) | 16) as u8;
}
}
if bio_integrity_flagged(bio, 32u32) {
unsafe {
(*scmd).prot_flags = (((*scmd).prot_flags as i32) | 2) as u8;
}
}
}
if dif != 3 {
// DIX/DIF Type 0, 1, 2
unsafe {
(*scmd).prot_flags = (((*scmd).prot_flags as i32) | 8) as u8;
}
if bio_integrity_flagged(bio, 64u32) {
unsafe {
(*scmd).prot_flags = (((*scmd).prot_flags as i32) | 4) as u8;
}
}
}
if dif != 0 {
// DIX/DIF Type 1, 2, 3
unsafe {
(*scmd).prot_flags = (((*scmd).prot_flags as i32) | 1) as u8;
}
if bio_integrity_flagged(bio, 4u32) {
protect = 96u32;
} else {
protect = 32u32;
}
// Disable target PI checking
// Enable target PI checking
}
scsi_set_prot_op(scmd, prot_op as u8);
scsi_set_prot_type(scmd, dif as u8);
let __v15: u32 = sd_prot_flag_mask(prot_op);
unsafe {
(*scmd).prot_flags = ((((*scmd).prot_flags as i32) as u32) & __v15) as u8;
}
protect as u8
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_uninit_command")]
extern "C" fn sd_uninit_command(cmd: *mut scsi_cmnd) {
let rq: *mut request = scsi_cmd_to_rq(cmd);
let sdp: *mut scsi_device = unsafe { (*cmd).device };
if (unsafe { (*rq).rq_flags }) & 4096u32 == 0 {
return;
}
if ((unsafe { (*sdp).sector_size }) as u64) > _AC!() << 12i32 {
unsafe { mempool_free((*rq).__slate_anon_25.special_vec.bv_page as *mut (), sd_large_page_pool) };
} else {
unsafe { mempool_free((*rq).__slate_anon_25.special_vec.bv_page as *mut (), sd_page_pool) };
}
let __v2: u32 = (unsafe { (*rq).rq_flags }) & 4294963199u32;
unsafe {
(*rq).rq_flags = __v2;
}
}

fn sd_set_special_bvec(cmd: *mut scsi_cmnd, data_len: u32) -> *mut () {
let mut page: *mut page = unsafe { core::mem::zeroed() };
let rq: *mut request = scsi_cmd_to_rq(cmd);
let sdp: *mut scsi_device = unsafe { (*cmd).device };
let sector_size: u32 = unsafe { (*sdp).sector_size };
let nr_pages: u32 = ((sector_size as u64).wrapping_add(4096u64).wrapping_sub(1) / 4096u64) as u32;
let mut n: i32 = 0;
if (sector_size as u64) > _AC!() << 12i32 {
page = alloc_hooks!(unsafe { mempool_alloc_noprof(sd_large_page_pool, 2080u32) }) as *mut page;
} else {
page = alloc_hooks!(unsafe { mempool_alloc_noprof(sd_page_pool, 2080u32) }) as *mut page;
}
if page.is_null() {
return core::ptr::null_mut::<()>();
}
while (n as u32) < nr_pages {
clear_highpage(page.wrapping_offset(n as isize));
n = n.wrapping_add(1);
}
bvec_set_page(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_25.special_vec) }, page, data_len, 0u32);
unsafe {
(*rq).rq_flags = (*rq).rq_flags | 4096u32;
}
bvec_virt(unsafe { core::ptr::addr_of_mut!((*rq).__slate_anon_25.special_vec) })
}

fn sd_setup_unmap_cmnd(cmd: *mut scsi_cmnd) -> u8 {
let sdp: *mut scsi_device = unsafe { (*cmd).device };
let rq: *mut request = scsi_cmd_to_rq(cmd);
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*(*rq).q).disk });
let lba: u64 = sectors_to_logical(sdp, blk_rq_pos(rq as *const request));
let nr_blocks: u32 = sectors_to_logical(sdp, blk_rq_sectors(rq as *const request) as u64) as u32;
let data_len: u32 = 24;
let mut ret: u8 = 0;
let buf: *mut u8 = sd_set_special_bvec(cmd, data_len) as *mut u8;
if buf.is_null() {
return 9;
}
unsafe {
(*cmd).cmd_len = 10;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 66;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(8 as isize) = 24;
}
put_unaligned_be16(22u16, buf.wrapping_offset(0 as isize) as *mut ());
put_unaligned_be16(16u16, buf.wrapping_offset(2 as isize) as *mut ());
put_unaligned_be64(lba, buf.wrapping_offset(8 as isize) as *mut ());
put_unaligned_be32(nr_blocks, buf.wrapping_offset(16 as isize) as *mut ());
unsafe {
(*cmd).allowed = (*sdkp).max_retries;
}
unsafe {
(*cmd).transfersize = data_len;
}
unsafe {
(*rq).timeout = 30000u32;
}
ret = unsafe { scsi_alloc_sgtables(cmd) };
if (ret as i32) != 0 {
sd_uninit_command(cmd);
}
ret
}

fn sd_config_atomic(sdkp: *mut scsi_disk, lim: *mut queue_limits) {
let logical_block_size: u32 = unsafe { (*(*sdkp).device).sector_size };
let mut physical_block_size_sectors: u32 = 0;
let mut max_atomic: u32 = 0;
let mut unit_min: u32 = 0;
let mut unit_max: u32 = 0;
if (unsafe { (*sdkp).max_atomic }) == 0 && (unsafe { (*sdkp).max_atomic_with_boundary }) == 0 || ((unsafe { (*sdkp).protection_type }) as i32) == 2 {
return;
}
physical_block_size_sectors = (unsafe { (*sdkp).physical_block_size }) / unsafe { (*(*sdkp).device).sector_size };
unit_min = rounddown_pow_of_two!(if (unsafe { (*sdkp).atomic_granularity }) != 0 { unsafe { (*sdkp).atomic_granularity } } else { physical_block_size_sectors }) as u32;
// Only use atomic boundary when we have the odd scenario of
// sdkp->max_atomic == 0, which the spec does permit.
if (unsafe { (*sdkp).max_atomic }) != 0 {
max_atomic = unsafe { (*sdkp).max_atomic };
unit_max = rounddown_pow_of_two!(unsafe { (*sdkp).max_atomic }) as u32;
unsafe {
(*sdkp).__slate_bits_0.__set_use_atomic_write_boundary(0u32);
}
} else {
max_atomic = unsafe { (*sdkp).max_atomic_with_boundary };
unit_max = rounddown_pow_of_two!(unsafe { (*sdkp).max_atomic_boundary }) as u32;
unsafe {
(*sdkp).__slate_bits_0.__set_use_atomic_write_boundary(1u32);
}
}
// Ensure compliance with granularity and alignment. For now, keep it
// simple and just don't support atomic writes for values mismatched
// with max_{boundary}atomic, physical block size, and
// atomic_granularity itself.
//
// We're really being distrustful by checking unit_max also...
if (unsafe { (*sdkp).atomic_granularity }) > 1 {
if unit_min > 1 && unit_min % unsafe { (*sdkp).atomic_granularity } != 0 {
return;
}
if unit_max > 1 && unit_max % unsafe { (*sdkp).atomic_granularity } != 0 {
return;
}
}
if (unsafe { (*sdkp).atomic_alignment }) > 1 {
if unit_min > 1 && unit_min % unsafe { (*sdkp).atomic_alignment } != 0 {
return;
}
if unit_max > 1 && unit_max % unsafe { (*sdkp).atomic_alignment } != 0 {
return;
}
}
unsafe {
(*lim).atomic_write_hw_max = max_atomic.wrapping_mul(logical_block_size);
}
unsafe {
(*lim).atomic_write_hw_boundary = 0;
}
unsafe {
(*lim).atomic_write_hw_unit_min = unit_min.wrapping_mul(logical_block_size);
}
unsafe {
(*lim).atomic_write_hw_unit_max = unit_max.wrapping_mul(logical_block_size);
}
unsafe {
(*lim).features = (*lim).features | 16384u32;
}
}

fn sd_setup_write_same16_cmnd(cmd: *mut scsi_cmnd, unmap: bool) -> u8 {
let sdp: *mut scsi_device = unsafe { (*cmd).device };
let rq: *mut request = scsi_cmd_to_rq(cmd);
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*(*rq).q).disk });
let lba: u64 = sectors_to_logical(sdp, blk_rq_pos(rq as *const request));
let nr_blocks: u32 = sectors_to_logical(sdp, blk_rq_sectors(rq as *const request) as u64) as u32;
let data_len: u32 = unsafe { (*sdp).sector_size };
let mut ret: u8 = 0;
if sd_set_special_bvec(cmd, data_len).is_null() {
return 9;
}
unsafe {
(*cmd).cmd_len = 16;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 147;
}
if unmap {
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = 8;
}
}
// UNMAP
put_unaligned_be64(lba, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(2 as isize) as *mut ());
put_unaligned_be32(nr_blocks, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(10 as isize) as *mut ());
unsafe {
(*cmd).allowed = (*sdkp).max_retries;
}
unsafe {
(*cmd).transfersize = data_len;
}
unsafe {
(*rq).timeout = (if unmap { 30000i32 } else { 120000i32 }) as u32;
}
ret = unsafe { scsi_alloc_sgtables(cmd) };
if (ret as i32) != 0 {
sd_uninit_command(cmd);
}
ret
}

fn sd_setup_write_same10_cmnd(cmd: *mut scsi_cmnd, unmap: bool) -> u8 {
let sdp: *mut scsi_device = unsafe { (*cmd).device };
let rq: *mut request = scsi_cmd_to_rq(cmd);
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*(*rq).q).disk });
let lba: u64 = sectors_to_logical(sdp, blk_rq_pos(rq as *const request));
let nr_blocks: u32 = sectors_to_logical(sdp, blk_rq_sectors(rq as *const request) as u64) as u32;
let data_len: u32 = unsafe { (*sdp).sector_size };
let mut ret: u8 = 0;
if sd_set_special_bvec(cmd, data_len).is_null() {
return 9;
}
unsafe {
(*cmd).cmd_len = 10;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 65;
}
if unmap {
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = 8;
}
}
// UNMAP
put_unaligned_be32(lba as u32, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(2 as isize) as *mut ());
put_unaligned_be16(nr_blocks as u16, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(7 as isize) as *mut ());
unsafe {
(*cmd).allowed = (*sdkp).max_retries;
}
unsafe {
(*cmd).transfersize = data_len;
}
unsafe {
(*rq).timeout = (if unmap { 30000i32 } else { 120000i32 }) as u32;
}
ret = unsafe { scsi_alloc_sgtables(cmd) };
if (ret as i32) != 0 {
sd_uninit_command(cmd);
}
ret
}

fn sd_setup_write_zeroes_cmnd(cmd: *mut scsi_cmnd) -> u8 {
let rq: *mut request = scsi_cmd_to_rq(cmd);
let sdp: *mut scsi_device = unsafe { (*cmd).device };
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*(*rq).q).disk });
let lba: u64 = sectors_to_logical(sdp, blk_rq_pos(rq as *const request));
let nr_blocks: u32 = sectors_to_logical(sdp, blk_rq_sectors(rq as *const request) as u64) as u32;
if (unsafe { (*rq).cmd_flags }) & 268435456u32 == 0 {
match (unsafe { (*sdkp).zeroing_mode }) as i32 {
2 => {
return sd_setup_write_same16_cmnd(cmd, true);
}
3 => {
return sd_setup_write_same10_cmnd(cmd, true);
}
_ => {
}
}
}
if ((unsafe { (*sdp).__slate_bits_0.__get_no_write_same() }) as i32) != 0 {
unsafe {
(*rq).rq_flags = (*rq).rq_flags | 128u32;
}
return 5;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_ws16() }) as i32) != 0 || lba > 4294967295 || nr_blocks > 65535 {
return sd_setup_write_same16_cmnd(cmd, false);
}
sd_setup_write_same10_cmnd(cmd, false)
}

fn sd_disable_write_same(sdkp: *mut scsi_disk) {
unsafe {
(*(*sdkp).device).__slate_bits_0.__set_no_write_same(1u32);
}
unsafe {
(*sdkp).max_ws_blocks = 0;
}
blk_queue_disable_write_zeroes(unsafe { (*(*sdkp).disk).queue });
}

fn sd_setup_flush_cmnd(cmd: *mut scsi_cmnd) -> u8 {
let rq: *mut request = scsi_cmd_to_rq(cmd);
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*(*rq).q).disk });
// flush requests don't perform I/O, zero the S/G table
unsafe { memset(core::ptr::addr_of_mut!((*cmd).sdb) as *mut (), 0i32, core::mem::size_of::<scsi_data_buffer>() as u64) };
if ((unsafe { (*(*cmd).device).__slate_bits_0.__get_use_16_for_sync() }) as i32) != 0 {
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 145;
}
unsafe {
(*cmd).cmd_len = 16;
}
} else {
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 53;
}
unsafe {
(*cmd).cmd_len = 10;
}
}
unsafe {
(*cmd).transfersize = 0;
}
unsafe {
(*cmd).allowed = (*sdkp).max_retries;
}
unsafe {
(*rq).timeout = unsafe { (*(*rq).q).rq_timeout }.wrapping_mul(2);
}
0
}

// sd_group_number() - Compute the GROUP NUMBER field
// @cmd: SCSI command for which to compute the value of the six-bit GROUP NUMBER
// 	field.
//
// From SBC-5 r05 (https://www.t10.org/cgi-bin/ac.pl?t=f&f=sbc5r05.pdf):
// 0: no relative lifetime.
// 1: shortest relative lifetime.
// 2: second shortest relative lifetime.
// 3 - 0x3d: intermediate relative lifetimes.
// 0x3e: second longest relative lifetime.
// 0x3f: longest relative lifetime.
fn sd_group_number(cmd: *mut scsi_cmnd) -> u8 {
let rq: *const request = scsi_cmd_to_rq(cmd) as *const request;
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*(*rq).q).disk });
if ((unsafe { (*sdkp).__slate_bits_0.__get_rscs() }) as i32) == 0 {
return 0;
}
let mut __v0: u32;
let __UNIQUE_ID_x__641: u32 = ((unsafe { (*(*rq).bio).bi_write_hint }) as u8) as u32;
let __UNIQUE_ID_y__642: u32 = (unsafe { (*sdkp).permanent_stream_count }) as u32;
let __UNIQUE_ID_z__643: u32 = 63;
loop {
if __types_ok3__2!(__UNIQUE_ID_x__641, __UNIQUE_ID_y__642, __UNIQUE_ID_z__643) == 0 {
unsafe { __compiletime_assert_644() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__641 < if __UNIQUE_ID_y__642 < __UNIQUE_ID_z__643 { __UNIQUE_ID_y__642 } else { __UNIQUE_ID_z__643 } { __UNIQUE_ID_x__641 } else { if __UNIQUE_ID_y__642 < __UNIQUE_ID_z__643 { __UNIQUE_ID_y__642 } else { __UNIQUE_ID_z__643 } };
__v0 as u8
}

fn sd_setup_rw32_cmnd(cmd: *mut scsi_cmnd, write: bool, lba: u64, nr_blocks: u32, flags: u8, dld: u32) -> u8 {
unsafe {
(*cmd).cmd_len = 32;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 127;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(6 as isize) = sd_group_number(cmd);
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(7 as isize) = 24;
}
// Additional CDB len
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(9 as isize) = (if write { 11i32 } else { 9i32 }) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(10 as isize) = flags;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(11 as isize) = (dld & 7) as u8;
}
put_unaligned_be64(lba, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(12 as isize) as *mut ());
put_unaligned_be32(lba as u32, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(20 as isize) as *mut ()); // Expected Indirect LBA
put_unaligned_be32(nr_blocks, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(28 as isize) as *mut ());
0
}

fn sd_setup_rw16_cmnd(cmd: *mut scsi_cmnd, write: bool, lba: u64, nr_blocks: u32, flags: u8, dld: u32) -> u8 {
unsafe {
(*cmd).cmd_len = 16;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = (if write { 138i32 } else { 136i32 }) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = (((flags as i32) as u32) | dld >> 2i32 & 1) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(14 as isize) = ((dld & 3) << 6i32 | ((sd_group_number(cmd) as i32) as u32)) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(15 as isize) = 0;
}
put_unaligned_be64(lba, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(2 as isize) as *mut ());
put_unaligned_be32(nr_blocks, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(10 as isize) as *mut ());
0
}

fn sd_setup_rw10_cmnd(cmd: *mut scsi_cmnd, write: bool, lba: u64, nr_blocks: u32, flags: u8) -> u8 {
unsafe {
(*cmd).cmd_len = 10;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = (if write { 42i32 } else { 40i32 }) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = flags;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(6 as isize) = sd_group_number(cmd);
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(9 as isize) = 0;
}
put_unaligned_be32(lba as u32, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(2 as isize) as *mut ());
put_unaligned_be16(nr_blocks as u16, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(7 as isize) as *mut ());
0
}

fn sd_setup_rw6_cmnd(cmd: *mut scsi_cmnd, write: bool, lba: u64, nr_blocks: u32, flags: u8) -> u8 {
// Avoid that 0 blocks gets translated into 256 blocks.
let mut __v0: i64;
let __ret_warn_on: i32 = (nr_blocks == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("645: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 645b - ., 3; .popsection\n# {0}", const 645 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_sd_setup_rw6_cmnd_1, sym __slate_asm_string_sd_setup_rw6_cmnd_2, const 1354 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("646: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 646b - ., 4; .popsection\n# {0}", const 646 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if __v0 != 0 {
return 10;
}
if unlikely__2!((flags as i32) & 8) != 0 {
// This happens only if this drive failed 10byte rw
// command with ILLEGAL_REQUEST during operation and
// thus turned off use_10_for_rw.
unsafe { scmd_printk(c"\x013".as_ptr() as *const u8, cmd, c"FUA write on READ/WRITE(6) drive\n".as_ptr() as *const u8) };
return 10;
}
unsafe {
(*cmd).cmd_len = 6;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = (if write { 10i32 } else { 8i32 }) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = (lba >> 16i32 & 31) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) = (lba >> 8i32 & 255) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(3 as isize) = (lba & 255) as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) = nr_blocks as u8;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) = 0;
}
0
}

/// Check if a command has a duration limit set. If it does, and the target
/// device supports CDL and the feature is enabled, return the limit
/// descriptor index to use. Return 0 (no limit) otherwise.
fn sd_cdl_dld(sdkp: *mut scsi_disk, scmd: *mut scsi_cmnd) -> i32 {
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut hint: i32 = 0;
if ((unsafe { (*sdp).__slate_bits_0.__get_cdl_supported() }) as i32) == 0 || ((unsafe { (*sdp).__slate_bits_0.__get_cdl_enable() }) as i32) == 0 {
return 0;
}
// Use "no limit" if the request ioprio does not specify a duration
// limit hint.
hint = IOPRIO_PRIO_HINT!(req_get_ioprio(scsi_cmd_to_rq(scmd)));
if hint < 1 || hint > 7 {
return 0;
}
hint.wrapping_sub(1).wrapping_add(1)
}

fn sd_setup_atomic_cmnd(cmd: *mut scsi_cmnd, lba: u64, nr_blocks: u32, boundary: bool, flags: u8) -> u8 {
unsafe {
(*cmd).cmd_len = 16;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) = 156;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = flags;
}
put_unaligned_be64(lba, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(2 as isize) as *mut ());
put_unaligned_be16(nr_blocks as u16, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(12 as isize) as *mut ());
if boundary {
put_unaligned_be16(nr_blocks as u16, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(10 as isize) as *mut ());
} else {
put_unaligned_be16(0u16, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(10 as isize) as *mut ());
}
put_unaligned_be16(nr_blocks as u16, unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }.wrapping_offset(12 as isize) as *mut ());
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(14 as isize) = 0;
}
unsafe {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(15 as isize) = 0;
}
0
}

fn sd_setup_read_write_cmnd(cmd: *mut scsi_cmnd) -> u8 {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut dix: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut bool = core::ptr::addr_of_mut!(dix) as *mut bool;
let mut dif: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u32 = core::ptr::addr_of_mut!(dif) as *mut u32;
let mut ret: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u8 = core::ptr::addr_of_mut!(ret) as *mut u8;
let mut dld: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u32 = core::ptr::addr_of_mut!(dld) as *mut u32;
let mut fua: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u8 = core::ptr::addr_of_mut!(fua) as *mut u8;
let mut protect: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u8 = core::ptr::addr_of_mut!(protect) as *mut u8;
let mut write: bool = unsafe { core::mem::zeroed() };
let mut mask: u32 = unsafe { core::mem::zeroed() };
let mut nr_blocks: u32 = unsafe { core::mem::zeroed() };
let mut threshold: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(threshold) as *mut u64;
let mut lba: u64 = unsafe { core::mem::zeroed() };
let mut sdkp: *mut scsi_disk = unsafe { core::mem::zeroed() };
let mut sdp: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut rq: *mut request = unsafe { core::mem::zeroed() };
unsafe {
rq = scsi_cmd_to_rq(cmd);
sdp = (*cmd).device;
sdkp = scsi_disk((*(*rq).q).disk);
lba = sectors_to_logical(sdp, blk_rq_pos(rq as *const request));
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
nr_blocks = sectors_to_logical(sdp, blk_rq_sectors(rq as *const request) as u64) as u32;
mask = logical_to_sectors(sdp, 1u64).wrapping_sub(1) as u32;
write = rq_data_dir!(rq) == 1;
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_5 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
*__slate_slot_3 = scsi_alloc_sgtables(cmd);
if (*__slate_slot_3 as i32) != 0 {
return *__slate_slot_3;
} else {
*__slate_slot_3 = 10;
if scsi_device_online(sdp) == 0 || ((*sdp).__slate_bits_0.__get_changed() as i32) != 0 {
scmd_printk(c"\x013".as_ptr() as *const u8, cmd, c"device offline or changed\n".as_ptr() as *const u8);
} else {
if blk_rq_pos(rq as *const request).wrapping_add(blk_rq_sectors(rq as *const request) as u64) > get_capacity((*(*rq).q).disk) {
scmd_printk(c"\x013".as_ptr() as *const u8, cmd, c"access beyond end of device\n".as_ptr() as *const u8);
} else {
if blk_rq_pos(rq as *const request) & (mask as u64) != 0 {
__v0 = true;
} else {
__v0 = blk_rq_sectors(rq as *const request) & mask != 0;
}
if __v0 {
scmd_printk(c"\x013".as_ptr() as *const u8, cmd, c"request not aligned to the logical block size\n".as_ptr() as *const u8);
} else {
// Some SD card readers can't handle accesses which touch the
// last one or two logical blocks. Split accesses as needed.
*__slate_slot_10 = unsafe { (*sdkp).capacity }.wrapping_sub(8);
if unlikely!(((*sdp).__slate_bits_0.__get_last_sector_bug() as i32) != 0 && lba.wrapping_add(nr_blocks as u64) > *__slate_slot_10) != 0 {
if lba < *__slate_slot_10 {
// Access up to the threshold but not beyond
nr_blocks = (*__slate_slot_10).wrapping_sub(lba) as u32;
} else {
// Access only a single logical block
nr_blocks = 1;
}
}
*__slate_slot_5 = (if (*rq).cmd_flags & 131072u32 != 0 { 8i32 } else { 0i32 }) as u8;
*__slate_slot_1 = scsi_prot_sg_count(cmd) != 0;
*__slate_slot_2 = scsi_host_dif_capable((*(*cmd).device).host, (*sdkp).protection_type as u32);
*__slate_slot_4 = sd_cdl_dld(sdkp, cmd) as u32;
if *__slate_slot_2 != 0 || *__slate_slot_1 {
*__slate_slot_6 = sd_setup_protect_cmnd(cmd, *__slate_slot_1 as u32, *__slate_slot_2);
} else {
*__slate_slot_6 = 0;
}
if *__slate_slot_6 != 0 && ((*sdkp).protection_type as i32) == 2 {
*__slate_slot_3 = sd_setup_rw32_cmnd(cmd, write, lba, nr_blocks, ((*__slate_slot_6 as i32) | (*__slate_slot_5 as i32)) as u8, *__slate_slot_4);
} else {
if (*rq).cmd_flags & 134217728u32 != 0 {
*__slate_slot_3 = sd_setup_atomic_cmnd(cmd, lba, nr_blocks, ((*sdkp).__slate_bits_0.__get_use_atomic_write_boundary() as i32) != 0, ((*__slate_slot_6 as i32) | (*__slate_slot_5 as i32)) as u8);
} else {
if ((*sdp).__slate_bits_0.__get_use_16_for_rw() as i32) != 0 || nr_blocks > 65535 {
*__slate_slot_3 = sd_setup_rw16_cmnd(cmd, write, lba, nr_blocks, ((*__slate_slot_6 as i32) | (*__slate_slot_5 as i32)) as u8, *__slate_slot_4);
} else {
if nr_blocks > 255 || lba > 2097151 || ((*sdp).__slate_bits_0.__get_use_10_for_rw() as i32) != 0 || *__slate_slot_6 != 0 || ((*(*rq).bio).bi_write_hint as u8) != 0 {
*__slate_slot_3 = sd_setup_rw10_cmnd(cmd, write, lba, nr_blocks, ((*__slate_slot_6 as i32) | (*__slate_slot_5 as i32)) as u8);
} else {
*__slate_slot_3 = sd_setup_rw6_cmnd(cmd, write, lba, nr_blocks, ((*__slate_slot_6 as i32) | (*__slate_slot_5 as i32)) as u8);
}
}
}
}
if unlikely!((*__slate_slot_3 as i32) != 0) == 0 {
// We shouldn't disconnect in the middle of a sector, so with a dumb
// host adapter, it's safe to assume that we can at least transfer
// this many bytes between each connect / disconnect.
(*cmd).transfersize = (*sdp).sector_size;
(*cmd).underflow = nr_blocks << 9i32;
(*cmd).allowed = (*sdkp).max_retries;
(*cmd).sdb.length = nr_blocks.wrapping_mul((*sdp).sector_size);
SCSI_LOG_HLQUEUE!();
SCSI_LOG_HLQUEUE!();
// This indicates that the command is ready from our end to be queued.
return 0;
}
}
}
}
scsi_free_sgtables(cmd);
return *__slate_slot_3;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_init_command")]
extern "C" fn sd_init_command(cmd: *mut scsi_cmnd) -> u8 {
let rq: *mut request = scsi_cmd_to_rq(cmd);
match req_op(rq as *const request) as u32 {
3 => {
match (unsafe { (*scsi_disk((*(*rq).q).disk)).provisioning_mode }) as i32 {
1 => {
return sd_setup_unmap_cmnd(cmd);
}
2 => {
return sd_setup_write_same16_cmnd(cmd, true);
}
3 => {
return sd_setup_write_same10_cmnd(cmd, true);
}
4 => {
return sd_setup_write_same10_cmnd(cmd, false);
}
_ => {
return 5;
}
}
return sd_setup_write_zeroes_cmnd(cmd);
}
9 => {
return sd_setup_write_zeroes_cmnd(cmd);
}
2 => {
return sd_setup_flush_cmnd(cmd);
}
0 | 1 => {
return sd_setup_read_write_cmnd(cmd);
}
17 => {
return sd_zbc_setup_zone_mgmt_cmnd(cmd, 4u8, false);
}
19 => {
return sd_zbc_setup_zone_mgmt_cmnd(cmd, 4u8, true);
}
11 => {
return sd_zbc_setup_zone_mgmt_cmnd(cmd, 3u8, false);
}
13 => {
return sd_zbc_setup_zone_mgmt_cmnd(cmd, 1u8, false);
}
15 => {
return sd_zbc_setup_zone_mgmt_cmnd(cmd, 2u8, false);
}
_ => {
let mut __v2: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("647: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 647b - ., 3; .popsection\n# {0}", const 647 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_sd_init_command_3, sym __slate_asm_string_sd_init_command_4, const 1575 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("648: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 648b - ., 4; .popsection\n# {0}", const 648 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
return 1;
}
}
unsafe { core::mem::zeroed() }
}

fn sd_need_revalidate(disk: *mut gendisk, sdkp: *mut scsi_disk) -> bool {
if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_removable() }) as i32) != 0 || (unsafe { (*sdkp).write_prot }) != 0 {
if unsafe { disk_check_media_change(disk) } {
return true;
}
}
// Force a full rescan after ioctl(BLKRRPART).  While the disk state has
// nothing to do with partitions, BLKRRPART is used to force a full
// revalidate after things like a format for historical reasons.
test_bit!(0i32, unsafe { core::ptr::addr_of_mut!((*disk).state) }) != 0
}

/// sd_open - open a scsi disk device
/// @disk: disk to open
/// @mode: open mode
///
/// Returns 0 if successful. Returns a negated errno value in case
/// of error.
///
/// Note: This can be called from a user context (e.g. fsck(1) )
/// or from within the kernel (e.g. as a result of a mount(1) ).
/// In the latter case @inode and @filp carry an abridged amount
/// of information as noted above.
///
/// Locking: called with disk->open_mutex held.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_open")]
extern "C" fn sd_open(disk: *mut gendisk, mode: u32) -> i32 {
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut sdkp: *mut scsi_disk = unsafe { core::mem::zeroed() };
unsafe {
sdkp = scsi_disk(disk);
sdev = (*sdkp).device;
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if scsi_device_get(sdev) != 0 {
return 6i32.wrapping_neg();
} else {
SCSI_LOG_HLQUEUE!();
// If the device is in error recovery, wait until it is done.
// If the device is offline, then disallow any access to it.
*__slate_slot_0 = 6i32.wrapping_neg();
if scsi_block_when_processing_errors(sdev) != 0 {
if sd_need_revalidate(disk, sdkp) {
sd_revalidate_disk(disk);
}
// If the drive is empty, just let the open fail.
*__slate_slot_0 = 123i32.wrapping_neg();
if !(((*sdev).__slate_bits_0.__get_removable() as i32) != 0 && (*sdkp).media_present == 0 && mode & 8u32 == 0) {
// If the device has the write protect tab set, have the open fail
// if the user expects to be able to write to the thing.
*__slate_slot_0 = 30i32.wrapping_neg();
if !((*sdkp).write_prot != 0 && mode & 2u32 != 0) {
// It is possible that the disk changing stuff resulted in
// the device being taken offline.  If this is the case,
// report this to the user, and don't pretend that the
// open actually succeeded.
*__slate_slot_0 = 6i32.wrapping_neg();
if scsi_device_online(sdev) != 0 {
if atomic_inc_return(core::ptr::addr_of_mut!((*sdkp).openers)) == 1 && ((*sdev).__slate_bits_0.__get_removable() as i32) != 0 {
if scsi_block_when_processing_errors(sdev) != 0 {
scsi_set_medium_removal(sdev, 1u8);
}
}
return 0;
}
}
}
}
scsi_device_put(sdev);
return *__slate_slot_0;
}
}
}

/// sd_release - invoked when the (last) close(2) is called on this
/// scsi disk.
/// @disk: disk to release
///
/// Returns 0.
///
/// Note: may block (uninterruptible) if error recovery is underway
/// on this disk.
///
/// Locking: called with disk->open_mutex held.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_release")]
extern "C" fn sd_release(disk: *mut gendisk) {
let sdkp: *mut scsi_disk = scsi_disk(disk);
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
SCSI_LOG_HLQUEUE!();
if atomic_dec_return(unsafe { core::ptr::addr_of_mut!((*sdkp).openers) }) == 0 && ((unsafe { (*sdev).__slate_bits_0.__get_removable() }) as i32) != 0 {
if (unsafe { scsi_block_when_processing_errors(sdev) }) != 0 {
unsafe { scsi_set_medium_removal(sdev, 0u8) };
}
}
unsafe { scsi_device_put(sdev) };
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_getgeo")]
extern "C" fn sd_getgeo(disk: *mut gendisk, geo: *mut hd_geometry) -> i32 {
let sdkp: *mut scsi_disk = scsi_disk(disk);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let host: *mut Scsi_Host = unsafe { (*sdp).host };
let capacity: u64 = logical_to_sectors(sdp, unsafe { (*sdkp).capacity });
let mut diskinfo: __SlateAlign16<[i32; 4]> = __SlateAlign16([0i32; 4]);
// default to most commonly used values
diskinfo.0[0] = 64; // 1 << 6
diskinfo.0[1] = 32; // 1 << 5
diskinfo.0[2] = (capacity >> 11i32) as i32;
// override with calculated, extended default, or driver values
if (unsafe { (*(*host).hostt).bios_param }) != None {
unsafe { unsafe { (*(*host).hostt).bios_param }.unwrap()(sdp, disk, capacity, diskinfo.0.as_mut_ptr() as *mut i32) };
} else {
unsafe { scsicam_bios_param(disk, capacity, diskinfo.0.as_mut_ptr() as *mut i32) };
}
unsafe {
(*geo).heads = diskinfo.0[0] as u8;
}
unsafe {
(*geo).sectors = diskinfo.0[1] as u8;
}
unsafe {
(*geo).cylinders = diskinfo.0[2] as u16;
}
0
}

/// sd_ioctl - process an ioctl
/// @bdev: target block device
/// @mode: open mode
/// @cmd: ioctl command number
/// @arg: this is third argument given to ioctl(2) system call.
/// Often contains a pointer.
///
/// Returns 0 if successful (some ioctls return positive numbers on
/// success as well). Returns a negated errno value in case of error.
///
/// Note: most ioctls are forward onto the block subsystem or further
/// down in the scsi subsystem.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_ioctl")]
extern "C" fn sd_ioctl(bdev: *mut block_device, mode: u32, cmd: u32, arg: u64) -> i32 {
let disk: *mut gendisk = unsafe { (*bdev).bd_disk };
let sdkp: *mut scsi_disk = scsi_disk(disk);
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let p: *mut () = arg as *mut ();
let mut error: i32 = 0;
SCSI_LOG_IOCTL!();
let __v1: bool = bdev_is_partition(bdev) && !unsafe { capable(17i32) };
if __v1 {
return 515i32.wrapping_neg();
}
// If we are in the middle of error recovery, don't let anyone
// else try and use this device.  Also, if error recovery fails, it
// may try and take the device offline, in which case all further
// access to the device is prohibited.
error = unsafe { scsi_ioctl_block_when_processing_errors(sdp, cmd as i32, mode & 8u32 != 0) };
if error != 0 {
return error;
}
if is_sed_ioctl(cmd) {
return sed_ioctl(unsafe { (*sdkp).opal_dev }, cmd, p);
}
unsafe { scsi_ioctl(sdp, mode & 2u32 != 0, cmd as i32, p) }
}

fn set_media_not_present(sdkp: *mut scsi_disk) {
if (unsafe { (*sdkp).media_present }) != 0 {
unsafe {
(*(*sdkp).device).__slate_bits_0.__set_changed(1u32);
}
}
if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_removable() }) as i32) != 0 {
unsafe {
(*sdkp).media_present = 0;
}
unsafe {
(*sdkp).capacity = 0;
}
}
}

fn media_not_present(sdkp: *mut scsi_disk, sshdr: *mut scsi_sense_hdr) -> i32 {
if !scsi_sense_valid(sshdr as *const scsi_sense_hdr) {
return 0;
}
// not invoked for commands that could return deferred errors
match (unsafe { (*sshdr).sense_key }) as i32 {
2 | 6 => {
if ((unsafe { (*sshdr).asc }) as i32) == 58 {
set_media_not_present(sdkp);
return 1;
}
// medium not present
}
_ => {
}
}
0
}

/// sd_check_events - check media events
/// @disk: kernel device descriptor
/// @clearing: disk events currently being cleared
///
/// Returns mask of DISK_EVENT_*.
///
/// Note: this function is invoked from the block subsystem.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_check_events")]
extern "C" fn sd_check_events(disk: *mut gendisk, clearing: u32) -> u32 {
let mut sshdr: core::mem::MaybeUninit<scsi_sense_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut scsi_sense_hdr = core::ptr::addr_of_mut!(sshdr) as *mut scsi_sense_hdr;
let mut disk_changed: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut bool = core::ptr::addr_of_mut!(disk_changed) as *mut bool;
let mut retval: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(retval) as *mut i32;
let mut sdp: core::mem::MaybeUninit<*mut scsi_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut scsi_device = core::ptr::addr_of_mut!(sdp) as *mut *mut scsi_device;
let mut sdkp: *mut scsi_disk = unsafe { core::mem::zeroed() };
unsafe {
sdkp = (*disk).private_data as *mut scsi_disk;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if sdkp.is_null() {
return 0;
} else {
'__join_0: {
*__slate_slot_3 = (*sdkp).device;
SCSI_LOG_HLQUEUE!();
// If the device is offline, don't send any commands - just pretend as
// if the command failed.  If the device ever comes back online, we
// can deal with it then.  It is only because of unrecoverable errors
// that we would ever take a device offline in the first place.
if scsi_device_online(*__slate_slot_3) == 0 {
set_media_not_present(sdkp);
} else {
// Using TEST_UNIT_READY enables differentiation between drive with
// no cartridge loaded - NOT READY, drive with changed cartridge -
// UNIT ATTENTION, or with same cartridge - GOOD STATUS.
//
// Drives that auto spin down. eg iomega jaz 1G, will be started
// by sd_spinup_disk() from sd_revalidate_disk(), which happens whenever
// sd_revalidate() is called.
if scsi_block_when_processing_errors(*__slate_slot_3) != 0 {
core::ptr::write(__slate_slot_0, {
let mut __t0: scsi_sense_hdr = core::mem::zeroed();
__t0.response_code = 0u8;
__t0
});
*__slate_slot_2 = scsi_test_unit_ready(*__slate_slot_3, 30000i32, (*sdkp).max_retries, core::ptr::addr_of_mut!(*__slate_slot_0));
// failed to execute TUR, assume media not present
if *__slate_slot_2 < 0 || host_byte!(*__slate_slot_2) != 0 {
set_media_not_present(sdkp);
break '__join_0;
} else {
if media_not_present(sdkp, core::ptr::addr_of_mut!(*__slate_slot_0)) != 0 {
break '__join_0;
}
}
}
// For removable scsi disk we have to recognise the presence
// of a disk in the drive.
if (*sdkp).media_present == 0 {
(*(*__slate_slot_3)).__slate_bits_0.__set_changed(1u32);
}
(*sdkp).media_present = 1;
}
}
*__slate_slot_1 = ((*(*__slate_slot_3)).__slate_bits_0.__get_changed() as i32) != 0;
// sdp->changed is set under the following conditions:
//
// 	Medium present state has changed in either direction.
// 	Device has indicated UNIT_ATTENTION.
(*(*__slate_slot_3)).__slate_bits_0.__set_changed(0u32);
return (if *__slate_slot_1 { 1i32 } else { 0i32 }) as u32;
}
}
}

fn sd_sync_cache(sdkp: *mut scsi_disk) -> i32 {
let mut res: i32 = 0;
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let timeout: i32 = unsafe { (*(*sdp).request_queue).rq_timeout }.wrapping_mul(2) as i32;
// Leave the rest of the command zero to indicate flush everything.
let mut cmd: __SlateAlign16<[u8; 16]> = __SlateAlign16({
let mut __t0: [u8; 16] = unsafe { core::mem::zeroed() };
__t0[0] = (if ((unsafe { (*sdp).__slate_bits_0.__get_use_16_for_sync() }) as i32) != 0 { 145i32 } else { 53i32 }) as u8;
__t0
});
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut failure_defs: __SlateAlign16<[scsi_failure; 2]> = __SlateAlign16([{
let mut __t1: scsi_failure = unsafe { core::mem::zeroed() };
__t1.result = 2147483647i32;
__t1.allowed = 3i8;
__t1
}, {
let mut __t2: scsi_failure = unsafe { core::mem::zeroed() };
__t2
}]);
let mut failures: scsi_failures = {
let mut __t3: scsi_failures = unsafe { core::mem::zeroed() };
__t3.failure_definitions = failure_defs.0.as_mut_ptr() as *mut scsi_failure;
__t3
};
let exec_args: scsi_exec_args = {
let mut __t4: scsi_exec_args = unsafe { core::mem::zeroed() };
__t4.sshdr = core::ptr::addr_of_mut!(sshdr);
__t4.req_flags = 4u32;
__t4.failures = core::ptr::addr_of_mut!(failures);
__t4
};
if scsi_device_online(sdp) == 0 {
return 19i32.wrapping_neg();
}
res = unsafe { scsi_execute_cmd(sdp, cmd.0.as_ptr() as *const u8, 34u32, core::ptr::null_mut::<()>(), 0u32, timeout, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
if res != 0 {
sd_print_result(sdkp as *const scsi_disk, c"Synchronize Cache(10) failed".as_ptr() as *const u8, res);
if res < 0 {
return res;
}
let __v0: bool = scsi_status_is_check_condition(res) != 0 && scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr);
if __v0 {
sd_print_sense_hdr(sdkp, core::ptr::addr_of_mut!(sshdr));
// we need to evaluate the error return
if (sshdr.asc as i32) == 58 || (sshdr.asc as i32) == 32 || (sshdr.asc as i32) == 116 && (sshdr.ascq as i32) == 113 {
return 0;
}
// medium not present
// invalid command
// drive is password locked
// this is no error here
// If a format is in progress or if the drive does not
// support sync, there is not much we can do because
// this is called during shutdown or suspend so just
// return success so those operations can proceed.
if (sshdr.asc as i32) == 4 && (sshdr.ascq as i32) == 4 || (sshdr.sense_key as i32) == 5 {
return 0;
}
}
match host_byte!(res) {
1 | 4 => {
// ignore errors due to racing a disconnection
return 0;
// signal the upper layer it might try again
}
2 | 11..=13 => {
return 16i32.wrapping_neg();
}
_ => {
return 5i32.wrapping_neg();
}
}
}
0
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_rescan")]
extern "C" fn sd_rescan(dev: *mut device) {
let sdkp: *mut scsi_disk = dev_get_drvdata(dev as *const device) as *mut scsi_disk;
sd_revalidate_disk(unsafe { (*sdkp).disk });
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_get_unique_id")]
extern "C" fn sd_get_unique_id(disk: *mut gendisk, id: *mut u8, r#type: u32) -> i32 {
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_rcu_649: *mut scsi_vpd = unsafe { core::mem::zeroed() };
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut d: core::mem::MaybeUninit<*const u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *const u8 = core::ptr::addr_of_mut!(d) as *mut *const u8;
let mut vpd: core::mem::MaybeUninit<*const scsi_vpd> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *const scsi_vpd = core::ptr::addr_of_mut!(vpd) as *mut *const scsi_vpd;
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
unsafe {
sdev = (*scsi_disk(disk)).device;
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
ret = 6i32.wrapping_neg();
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
rcu_read_lock();
while false {
}
let __v4: *mut scsi_vpd = core::ptr::read_volatile(core::ptr::addr_of_mut!((*sdev).vpd_pg83) as *const *mut scsi_vpd);
__UNIQUE_ID_rcu_649 = __v4;
loop {
if false {
if false {
__v2 = true;
} else {
__v2 = rcu_read_lock_held() != 0;
}
__v3 = !__v2;
} else {
__v3 = false;
}
if !__v3 {
break;
}
}
*__slate_slot_10 = __UNIQUE_ID_rcu_649 as *const scsi_vpd;
if !(*__slate_slot_10).is_null() {
ret = 22i32.wrapping_neg();
*__slate_slot_9 = (core::ptr::addr_of!((*(*__slate_slot_10)).data) as *const u8).wrapping_offset(4 as isize);
while *__slate_slot_9 < (core::ptr::addr_of!((*(*__slate_slot_10)).data) as *const u8).wrapping_offset((*(*__slate_slot_10)).len as isize) {
// we only care about designators with LU association
if (*(*__slate_slot_9).wrapping_offset(1 as isize) as i32) >> 4i32 & 3 == 0 {
if (((*(*__slate_slot_9).wrapping_offset(1 as isize) as i32) & 15) as u32) == (r#type as u32) {
// Only exit early if a 16-byte descriptor was found.  Otherwise
// keep looking as one with more entropy might still show up.
*__slate_slot_7 = *(*__slate_slot_9).wrapping_offset(3 as isize) as i32;
if !(*__slate_slot_7 != 8 && *__slate_slot_7 != 12 && *__slate_slot_7 != 16) {
ret = *__slate_slot_7;
memcpy(id as *mut (), (*__slate_slot_9).wrapping_offset(4 as isize) as *const (), *__slate_slot_7 as u64);
if *__slate_slot_7 == 16 {
break;
}
}
}
}
let __v1: *const u8 = *__slate_slot_9;
*__slate_slot_9 = __v1.wrapping_offset((*(*__slate_slot_9).wrapping_offset(3 as isize) as i32).wrapping_add(4) as isize);
}
}
rcu_read_unlock();
return ret;
}
}

fn sd_scsi_to_pr_err(sshdr: *mut scsi_sense_hdr, result: i32) -> i32 {
match host_byte!(result) {
2 | 14 | 20 => {
return 917504;
}
1 => {
return 65536;
}
15 => {
return 983040;
}
_ => {
}
}
match status_byte!(result) {
24 => {
return 24;
}
2 => {
if !scsi_sense_valid(sshdr as *const scsi_sense_hdr) {
return 2;
}
if ((unsafe { (*sshdr).sense_key }) as i32) == 5 && (((unsafe { (*sshdr).asc }) as i32) == 38 || ((unsafe { (*sshdr).asc }) as i32) == 36) {
return 22i32.wrapping_neg();
}
return 2;
}
_ => {
return 2;
}
}
unsafe { core::mem::zeroed() }
}

fn sd_pr_in_command(bdev: *mut block_device, sa: u8, data: *mut u8, data_len: i32) -> i32 {
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*bdev).bd_disk });
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut cmd: [u8; 10] = {
let mut __t0: [u8; 10] = unsafe { core::mem::zeroed() };
__t0[0] = 94u8;
__t0[1] = sa;
__t0
};
let mut failure_defs: __SlateAlign16<[scsi_failure; 2]> = __SlateAlign16([{
let mut __t1: scsi_failure = unsafe { core::mem::zeroed() };
__t1.result = 2i32;
__t1.sense = 6u8;
__t1.asc = 255u8;
__t1.ascq = 255u8;
__t1.allowed = 5i8;
__t1
}, {
let mut __t2: scsi_failure = unsafe { core::mem::zeroed() };
__t2
}]);
let mut failures: scsi_failures = {
let mut __t3: scsi_failures = unsafe { core::mem::zeroed() };
__t3.failure_definitions = failure_defs.0.as_mut_ptr() as *mut scsi_failure;
__t3
};
let exec_args: scsi_exec_args = {
let mut __t4: scsi_exec_args = unsafe { core::mem::zeroed() };
__t4.sshdr = core::ptr::addr_of_mut!(sshdr);
__t4.failures = core::ptr::addr_of_mut!(failures);
__t4
};
let mut result: i32 = 0;
put_unaligned_be16(data_len as u16, (cmd.as_mut_ptr() as *mut u8).wrapping_offset(7 as isize) as *mut ());
result = unsafe { scsi_execute_cmd(sdev, cmd.as_mut_ptr() as *const u8, 34u32, data as *mut (), data_len as u32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
let __v0: bool = scsi_status_is_check_condition(result) != 0 && scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr);
if __v0 {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, sdev as *const scsi_device, core::ptr::null::<u8>(), c"PR command failed: %d\n".as_ptr() as *const u8, result) };
unsafe { scsi_print_sense_hdr(sdev as *const scsi_device, core::ptr::null::<u8>(), core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr) };
}
if result <= 0 {
return result;
}
sd_scsi_to_pr_err(core::ptr::addr_of_mut!(sshdr), result)
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_pr_read_keys")]
extern "C" fn sd_pr_read_keys(bdev: *mut block_device, keys_info: *mut pr_keys) -> i32 {
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_y__652: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__651: u32 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
// Each reservation key takes 8 bytes and there is an 8-byte header
// before the reservation key list. The total size must fit into the
// 16-bit ALLOCATION LENGTH field.
let mut __v11: bool = unsafe { core::mem::zeroed() };
let mut data: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut u8 = core::ptr::addr_of_mut!(data) as *mut *mut u8;
let mut data_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(data_len) as *mut i32;
let mut num_keys: u32 = unsafe { core::mem::zeroed() };
let mut num_copy_keys: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(num_copy_keys) as *mut i32;
let mut data_offset: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(data_offset) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(result) as *mut i32;
unsafe {
{
let __slate_auto_bytes_2 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
num_keys = (*keys_info).num_keys;
let __slate_auto_bytes_6 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
if check_mul_overflow__2!(num_keys, 8i32, core::ptr::addr_of_mut!(*__slate_slot_13)) {
__v11 = true;
} else {
__v11 = check_add_overflow__2!(*__slate_slot_13, 8i32, core::ptr::addr_of_mut!(*__slate_slot_13));
}
}
if __v11 || *__slate_slot_13 > 65535i32 {
return 22i32.wrapping_neg();
} else {
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_7 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_8 = _kzalloc_noprof(*__slate_slot_13 as u64, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
alloc_tag_restore!();
} else {
*__slate_slot_8 = _kzalloc_noprof(*__slate_slot_13 as u64, 3264u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
}
*__slate_slot_12 = *__slate_slot_8 as *mut u8;
if (*__slate_slot_12).is_null() {
return 12i32.wrapping_neg();
} else {
*__slate_slot_18 = sd_pr_in_command(bdev, 0u8, *__slate_slot_12, *__slate_slot_13);
if *__slate_slot_18 == 0 {
(*keys_info).generation = get_unaligned_be32((*__slate_slot_12).wrapping_offset(0 as isize) as *const ());
(*keys_info).num_keys = get_unaligned_be32((*__slate_slot_12).wrapping_offset(4 as isize) as *const ()) / 8;
*__slate_slot_16 = 8;
__UNIQUE_ID_x__651 = num_keys;
__UNIQUE_ID_y__652 = (*keys_info).num_keys;
while false {
}
*__slate_slot_15 = (if __UNIQUE_ID_x__651 < __UNIQUE_ID_y__652 { __UNIQUE_ID_x__651 } else { __UNIQUE_ID_y__652 }) as i32;
*__slate_slot_17 = 0;
while *__slate_slot_17 < *__slate_slot_15 {
*(core::ptr::addr_of_mut!((*keys_info).keys) as *mut u64).wrapping_offset(*__slate_slot_17 as isize) = get_unaligned_be64((*__slate_slot_12).wrapping_offset(*__slate_slot_16 as isize) as *const ());
let __v3: i32 = *__slate_slot_16;
*__slate_slot_16 = __v3.wrapping_add(8);
let __v1: i32 = *__slate_slot_17;
*__slate_slot_17 = __v1.wrapping_add(1);
}
}
kfree(*__slate_slot_12 as *const ());
return *__slate_slot_18;
}
}
}
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_pr_read_reservation")]
extern "C" fn sd_pr_read_reservation(bdev: *mut block_device, rsv: *mut pr_held_reservation) -> i32 {
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*bdev).bd_disk });
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
let mut data: __SlateAlign16<[u8; 24]> = __SlateAlign16({
let mut __t0: [u8; 24] = unsafe { core::mem::zeroed() };
__t0
});
let result: i32 = sd_pr_in_command(bdev, 1u8, data.0.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 24]>() as i32);
let mut len: i32 = 0;
if result != 0 {
return result;
}
len = get_unaligned_be32((data.0.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) as *const ()) as i32;
if len == 0 {
return 0;
}
// Make sure we have at least the key and type
if len < 14 {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, sdev as *const scsi_device, core::ptr::null::<u8>(), c"READ RESERVATION failed due to short return buffer of %d bytes\n".as_ptr() as *const u8, len) };
return 22i32.wrapping_neg();
}
unsafe {
(*rsv).generation = get_unaligned_be32((data.0.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as *const ());
}
unsafe {
(*rsv).key = get_unaligned_be64((data.0.as_mut_ptr() as *mut u8).wrapping_offset(8 as isize) as *const ());
}
unsafe {
(*rsv).r#type = scsi_pr_type_to_block(((data.0[21] as i32) & 15) as u32);
}
0
}

fn sd_pr_out_command(bdev: *mut block_device, sa: u8, key: u64, sa_key: u64, r#type: u32, flags: u8) -> i32 {
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*bdev).bd_disk });
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut failure_defs: __SlateAlign16<[scsi_failure; 2]> = __SlateAlign16([{
let mut __t0: scsi_failure = unsafe { core::mem::zeroed() };
__t0.result = 2i32;
__t0.sense = 6u8;
__t0.asc = 255u8;
__t0.ascq = 255u8;
__t0.allowed = 5i8;
__t0
}, {
let mut __t1: scsi_failure = unsafe { core::mem::zeroed() };
__t1
}]);
let mut failures: scsi_failures = {
let mut __t2: scsi_failures = unsafe { core::mem::zeroed() };
__t2.failure_definitions = failure_defs.0.as_mut_ptr() as *mut scsi_failure;
__t2
};
let exec_args: scsi_exec_args = {
let mut __t3: scsi_exec_args = unsafe { core::mem::zeroed() };
__t3.sshdr = core::ptr::addr_of_mut!(sshdr);
__t3.failures = core::ptr::addr_of_mut!(failures);
__t3
};
let mut result: i32 = 0;
let mut cmd: __SlateAlign16<[u8; 16]> = __SlateAlign16({
let mut __t4: [u8; 16] = unsafe { core::mem::zeroed() };
__t4[0] = 0u8;
__t4
});
let mut data: __SlateAlign16<[u8; 24]> = __SlateAlign16({
let mut __t5: [u8; 24] = unsafe { core::mem::zeroed() };
__t5[0] = 0u8;
__t5
});
cmd.0[0] = 95;
cmd.0[1] = sa;
cmd.0[2] = (r#type as u32) as u8;
put_unaligned_be32(core::mem::size_of::<[u8; 24]>() as u32, (cmd.0.as_mut_ptr() as *mut u8).wrapping_offset(5 as isize) as *mut ());
put_unaligned_be64(key, (data.0.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as *mut ());
put_unaligned_be64(sa_key, (data.0.as_mut_ptr() as *mut u8).wrapping_offset(8 as isize) as *mut ());
data.0[20] = flags;
result = unsafe { scsi_execute_cmd(sdev, cmd.0.as_mut_ptr() as *const u8, 35u32, core::ptr::addr_of_mut!(data.0) as *mut (), core::mem::size_of::<[u8; 24]>() as u32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
let __v0: bool = scsi_status_is_check_condition(result) != 0 && scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr);
if __v0 {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, sdev as *const scsi_device, core::ptr::null::<u8>(), c"PR command failed: %d\n".as_ptr() as *const u8, result) };
unsafe { scsi_print_sense_hdr(sdev as *const scsi_device, core::ptr::null::<u8>(), core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr) };
}
if result <= 0 {
return result;
}
sd_scsi_to_pr_err(core::ptr::addr_of_mut!(sshdr), result)
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_pr_register")]
extern "C" fn sd_pr_register(bdev: *mut block_device, old_key: u64, new_key: u64, flags: u32) -> i32 {
if flags & 4294967294u32 != 0 {
return 95i32.wrapping_neg();
}
return sd_pr_out_command(bdev, (if flags & 1u32 != 0 { 6i32 } else { 0i32 }) as u8, old_key, new_key, 0u32, 1u8); // APTPL
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_pr_reserve")]
extern "C" fn sd_pr_reserve(bdev: *mut block_device, key: u64, r#type: u32, flags: u32) -> i32 {
if flags != 0 {
return 95i32.wrapping_neg();
}
sd_pr_out_command(bdev, 1u8, key, 0u64, unsafe { block_pr_type_to_scsi(r#type) }, 0u8)
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_pr_release")]
extern "C" fn sd_pr_release(bdev: *mut block_device, key: u64, r#type: u32) -> i32 {
sd_pr_out_command(bdev, 2u8, key, 0u64, unsafe { block_pr_type_to_scsi(r#type) }, 0u8)
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_pr_preempt")]
extern "C" fn sd_pr_preempt(bdev: *mut block_device, old_key: u64, new_key: u64, r#type: u32, abort: bool) -> i32 {
sd_pr_out_command(bdev, (if abort { 5i32 } else { 4i32 }) as u8, old_key, new_key, unsafe { block_pr_type_to_scsi(r#type) }, 0u8)
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_pr_clear")]
extern "C" fn sd_pr_clear(bdev: *mut block_device, key: u64) -> i32 {
sd_pr_out_command(bdev, 3u8, key, 0u64, 0u32, 0u8)
}

static sd_pr_ops: __SlateConst<pr_ops> = __SlateConst(pr_ops { pr_register: Some(sd_pr_register), pr_reserve: Some(sd_pr_reserve), pr_release: Some(sd_pr_release), pr_preempt: Some(sd_pr_preempt), pr_clear: Some(sd_pr_clear), pr_read_keys: Some(sd_pr_read_keys), pr_read_reservation: Some(sd_pr_read_reservation) });

#[unsafe(link_section = ".text.slate_distinct.sd.scsi_disk_free_disk")]
extern "C" fn scsi_disk_free_disk(disk: *mut gendisk) {
let sdkp: *mut scsi_disk = scsi_disk(disk);
unsafe { put_device(core::ptr::addr_of_mut!((*sdkp).disk_dev)) };
}

/// sd_eh_reset - reset error handling callback
/// @scmd:		sd-issued command that has failed
///
/// This function is called by the SCSI midlayer before starting
/// SCSI EH. When counting medium access failures we have to be
/// careful to register it only only once per device and SCSI EH run;
/// there might be several timed out commands which will cause the
/// 'max_medium_access_timeouts' counter to trigger after the first
/// SCSI EH run already and set the device to offline.
/// So this function resets the internal counter before starting SCSI EH.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_eh_reset")]
extern "C" fn sd_eh_reset(scmd: *mut scsi_cmnd) {
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*(*scsi_cmd_to_rq(scmd)).q).disk });
// New SCSI EH run, reset gate variable
unsafe {
(*sdkp).__slate_bits_0.__set_ignore_medium_access_errors(0u32);
}
}

/// sd_eh_action - error handling callback
/// @scmd:		sd-issued command that has failed
/// @eh_disp:	The recovery disposition suggested by the midlayer
///
/// This function is called by the SCSI midlayer upon completion of an
/// error test command (currently TEST UNIT READY). The result of sending
/// the eh command is passed in eh_disp.  We're looking for devices that
/// fail medium access commands but are OK with non access commands like
/// test unit ready (so wrongly see the device as having a successful
/// recovery)
#[unsafe(link_section = ".text.slate_distinct.sd.sd_eh_action")]
extern "C" fn sd_eh_action(scmd: *mut scsi_cmnd, eh_disp: i32) -> i32 {
let sdkp: *mut scsi_disk = scsi_disk(unsafe { (*(*scsi_cmd_to_rq(scmd)).q).disk });
let sdev: *mut scsi_device = unsafe { (*scmd).device };
let __v0: bool = scsi_device_online(sdev) == 0 || scsi_medium_access_command(scmd) == 0;
if __v0 || host_byte!(unsafe { (*scmd).result }) != 3 || eh_disp != 8194 {
return eh_disp;
}
// The device has timed out executing a medium access command.
// However, the TEST UNIT READY command sent during error
// handling completed successfully. Either the device is in the
// process of recovering or has it suffered an internal failure
// that prevents access to the storage medium.
if ((unsafe { (*sdkp).__slate_bits_0.__get_ignore_medium_access_errors() }) as i32) == 0 {
let __v2: u32 = unsafe { (*sdkp).medium_access_timed_out };
unsafe {
(*sdkp).medium_access_timed_out = __v2.wrapping_add(1);
}
unsafe {
(*sdkp).__slate_bits_0.__set_ignore_medium_access_errors(1u32);
}
}
// If the device keeps failing read/write commands but TEST UNIT
// READY always completes successfully we assume that medium
// access is no longer possible and take the device offline.
if (unsafe { (*sdkp).medium_access_timed_out }) >= unsafe { (*sdkp).max_medium_access_timeouts } {
unsafe { scmd_printk(c"\x013".as_ptr() as *const u8, scmd, c"Medium access timeout failure. Offlining disk!\n".as_ptr() as *const u8) };
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
unsafe { scsi_device_set_state(sdev, 6u32) };
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
return 8194;
}
eh_disp
}

fn sd_completed_bytes(scmd: *mut scsi_cmnd) -> u32 {
let req: *mut request = scsi_cmd_to_rq(scmd);
let sdev: *mut scsi_device = unsafe { (*scmd).device };
let mut transferred: u32 = 0;
let mut good_bytes: u32 = 0;
let mut start_lba: u64 = 0;
let mut end_lba: u64 = 0;
let mut bad_lba: u64 = 0;
// Some commands have a payload smaller than the device logical
// block size (e.g. INQUIRY on a 4K disk).
if scsi_bufflen(scmd) <= unsafe { (*sdev).sector_size } {
return 0;
}
// Check if we have a 'bad_lba' information
if !unsafe { scsi_get_sense_info_fld((*scmd).sense_buffer as *const u8, 96i32, core::ptr::addr_of_mut!(bad_lba)) } {
return 0;
}
// If the bad lba was reported incorrectly, we have no idea where
// the error is.
start_lba = sectors_to_logical(sdev, blk_rq_pos(req as *const request));
end_lba = start_lba.wrapping_add(bytes_to_logical(sdev, scsi_bufflen(scmd)));
if bad_lba < start_lba || bad_lba >= end_lba {
return 0;
}
// resid is optional but mostly filled in.  When it's unused,
// its value is zero, so we assume the whole buffer transferred
transferred = scsi_bufflen(scmd).wrapping_sub(scsi_get_resid(scmd));
// This computation should always be done in terms of the
// resolution of the device's medium.
good_bytes = logical_to_bytes(sdev, bad_lba.wrapping_sub(start_lba));
let mut __v0: u32;
let __UNIQUE_ID_x__654: u32 = good_bytes;
let __UNIQUE_ID_y__655: u32 = transferred;
loop {
if __types_ok!(__UNIQUE_ID_x__654, __UNIQUE_ID_y__655) == 0 {
unsafe { __compiletime_assert_656() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__654 < __UNIQUE_ID_y__655 { __UNIQUE_ID_x__654 } else { __UNIQUE_ID_y__655 };
__v0
}

// sd_done - bottom half handler: called when the lower level
// driver has completed (successfully or otherwise) a scsi command.
// @SCpnt: mid-level's per command structure.
//
// Note: potentially run from within an ISR. Must not block.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_done")]
extern "C" fn sd_done(SCpnt: *mut scsi_cmnd) -> i32 {
let mut __UNIQUE_ID_y__658: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_x__657: u32 = unsafe { core::mem::zeroed() };
let mut sense_deferred: i32 = unsafe { core::mem::zeroed() };
let mut sense_valid: i32 = unsafe { core::mem::zeroed() };
let mut sdkp: *mut scsi_disk = unsafe { core::mem::zeroed() };
let mut req: *mut request = unsafe { core::mem::zeroed() };
let mut sshdr: core::mem::MaybeUninit<scsi_sense_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut scsi_sense_hdr = core::ptr::addr_of_mut!(sshdr) as *mut scsi_sense_hdr;
let mut resid: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(resid) as *mut u32;
let mut sector_size: u32 = unsafe { core::mem::zeroed() };
let mut __v13: u32 = unsafe { core::mem::zeroed() };
let mut good_bytes: u32 = unsafe { core::mem::zeroed() };
let mut result: i32 = unsafe { core::mem::zeroed() };
unsafe {
result = (*SCpnt).result;
if result != 0 {
__v13 = 0;
} else {
__v13 = scsi_bufflen(SCpnt);
}
'__join_21: {
good_bytes = __v13;
sector_size = (*(*SCpnt).device).sector_size;
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
req = scsi_cmd_to_rq(SCpnt);
sdkp = scsi_disk((*(*req).q).disk);
sense_valid = 0;
sense_deferred = 0;
let __t3: u32 = req_op(req as *const request) as u32;
if __t3 != 3 {
if __t3 != 9 {
if __t3 != 17 {
if __t3 != 19 {
if __t3 != 11 {
if __t3 != 13 {
if __t3 != 15 {
*__slate_slot_11 = scsi_get_resid(SCpnt);
// In case of bogus fw or device, we could end up having
// an unaligned partial completion. Check this here and force
// alignment.
if *__slate_slot_11 & sector_size.wrapping_sub(1) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Unaligned partial completion (resid=%u, sector_sz=%u)\n".as_ptr() as *const u8, *__slate_slot_11, sector_size);
} else {
sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Unaligned partial completion (resid=%u, sector_sz=%u)\n".as_ptr() as *const u8, *__slate_slot_11, sector_size);
}
scsi_print_command(SCpnt);
__UNIQUE_ID_x__657 = scsi_bufflen(SCpnt);
__UNIQUE_ID_y__658 = round_up!(*__slate_slot_11, sector_size);
while false {
}
*__slate_slot_11 = if __UNIQUE_ID_x__657 < __UNIQUE_ID_y__658 { __UNIQUE_ID_x__657 } else { __UNIQUE_ID_y__658 };
scsi_set_resid(SCpnt, *__slate_slot_11);
break '__join_21;
} else {
break '__join_21;
}
}
}
}
}
}
}
}
if result == 0 {
good_bytes = blk_rq_bytes(req as *const request);
scsi_set_resid(SCpnt, 0u32);
} else {
good_bytes = 0;
scsi_set_resid(SCpnt, blk_rq_bytes(req as *const request));
}
}
if result != 0 {
sense_valid = scsi_command_normalize_sense(SCpnt as *const scsi_cmnd, core::ptr::addr_of_mut!(*__slate_slot_10)) as i32;
if sense_valid != 0 {
sense_deferred = scsi_sense_is_deferred(&(*(core::ptr::addr_of_mut!(*__slate_slot_10) as *const scsi_sense_hdr))) as i32;
}
}
'__join_2: {
(*sdkp).medium_access_timed_out = 0;
if !(scsi_status_is_check_condition(result) == 0 || sense_valid == 0 || sense_deferred != 0) {
let __t2: i32 = (*__slate_slot_10).sense_key as i32;
if __t2 != 4 {
if __t2 != 3 {
if __t2 == 1 {
good_bytes = scsi_bufflen(SCpnt);
break '__join_2;
} else {
if __t2 == 0 {
(*SCpnt).result = 0;
// This indicates a false check condition, so ignore it.  An
// unknown amount of data was transferred so treat it as an
// error.
memset((*SCpnt).sense_buffer as *mut (), 0i32, 96u64);
break '__join_2;
} else {
if __t2 == 11 {
if ((*__slate_slot_10).asc as i32) == 16 {
good_bytes = sd_completed_bytes(SCpnt);
}
// DIF: Target detected corruption
break '__join_2;
} else {
if __t2 == 5 {
let __t1: i32 = (*__slate_slot_10).asc as i32;
if __t1 == 16 {
good_bytes = sd_completed_bytes(SCpnt); // DIX: Host detected corruption
break '__join_2;
} else {
if __t1 != 32 {
if __t1 != 36 {
break '__join_2;
}
}
}
'__join_3: {
let __t0: i32 = *((*SCpnt).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32;
if __t0 == 66 {
sd_disable_discard(sdkp);
break '__join_3;
} else {
if __t0 != 147 {
if __t0 != 65 {
break '__join_3;
}
}
}
if (*((*SCpnt).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) & 8 != 0 {
// UNMAP
sd_disable_discard(sdkp);
} else {
sd_disable_write_same(sdkp);
(*req).rq_flags = (*req).rq_flags | 128u32;
}
}
// INVALID COMMAND OPCODE
// INVALID FIELD IN CDB
break '__join_2;
} else {
break '__join_2;
}
}
}
}
}
}
good_bytes = sd_completed_bytes(SCpnt);
}
}
if ((*(*sdkp).device).r#type as i32) == 20 {
good_bytes = sd_zbc_complete(SCpnt, good_bytes, &(*core::ptr::addr_of_mut!(*__slate_slot_10)));
}
SCSI_LOG_HLCOMPLETE!();
return good_bytes as i32;
}
}

/// spinup disk - called only in sd_revalidate_disk()
fn sd_spinup_disk(sdkp: *mut scsi_disk) {
let mut spintime_expire: u64 = 0;
let mut the_result: i32 = 0;
let mut spintime: i32 = 0;
let mut sense_valid: i32 = 0;
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut failure_defs: __SlateAlign16<[scsi_failure; 4]> = __SlateAlign16([{
let mut __t0: scsi_failure = unsafe { core::mem::zeroed() };
__t0.result = 2i32;
__t0.sense = 6u8;
__t0.asc = 58u8;
__t0.ascq = 255u8;
__t0
}, {
let mut __t1: scsi_failure = unsafe { core::mem::zeroed() };
__t1.result = 2i32;
__t1.sense = 2u8;
__t1.asc = 58u8;
__t1.ascq = 255u8;
__t1
}, {
let mut __t2: scsi_failure = unsafe { core::mem::zeroed() };
__t2.result = 255i32;
__t2.allowed = 3i8;
__t2
}, {
let mut __t3: scsi_failure = unsafe { core::mem::zeroed() };
__t3
}]);
// Do not retry Medium Not Present
// Retry when scsi_status_is_good would return false 3 times
let mut failures: scsi_failures = {
let mut __t4: scsi_failures = unsafe { core::mem::zeroed() };
__t4.failure_definitions = failure_defs.0.as_mut_ptr() as *mut scsi_failure;
__t4
};
let exec_args: scsi_exec_args = {
let mut __t5: scsi_exec_args = unsafe { core::mem::zeroed() };
__t5.sshdr = core::ptr::addr_of_mut!(sshdr);
__t5.failures = core::ptr::addr_of_mut!(failures);
__t5
};
// Spin up drives, as required.  Only do this at boot time
//
// Spinup needs to be done for module loads too.
'__slate_break_1: loop {
let media_was_present: bool = (unsafe { (*sdkp).media_present }) != 0;
unsafe { scsi_failures_reset_retries(core::ptr::addr_of_mut!(failures)) };
the_result = unsafe { scsi_execute_cmd((*sdkp).device, core::ptr::addr_of!(cmd__2490) as *const u8, 34u32, core::ptr::null_mut::<()>(), 0u32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
if the_result > 0 {
// If the drive has indicated to us that it doesn't
// have any media in it, don't bother with any more
// polling.
if media_not_present(sdkp, core::ptr::addr_of_mut!(sshdr)) != 0 {
if media_was_present {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Media removed, stopped polling\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Media removed, stopped polling\n".as_ptr() as *const u8) };
}
}
return;
}
sense_valid = scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr) as i32;
}
if scsi_status_is_check_condition(the_result) == 0 {
// no sense, TUR either succeeded or failed
// with a status error
let __v0: bool = spintime == 0 && !scsi_status_is_good(the_result);
if __v0 {
sd_print_result(sdkp as *const scsi_disk, c"Test Unit Ready failed".as_ptr() as *const u8, the_result);
}
break '__slate_break_1;
}
// The device does not want the automatic start to be issued.
if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_no_start_on_add() }) as i32) != 0 {
break '__slate_break_1;
}
if sense_valid != 0 && (sshdr.sense_key as i32) == 2 {
if (sshdr.asc as i32) == 4 && (sshdr.ascq as i32) == 3 {
break '__slate_break_1;
}
// manual intervention required
if (sshdr.asc as i32) == 4 && (sshdr.ascq as i32) == 11 {
break '__slate_break_1;
}
// standby
if (sshdr.asc as i32) == 4 && (sshdr.ascq as i32) == 12 {
break '__slate_break_1;
}
// unavailable
if (sshdr.asc as i32) == 4 && (sshdr.ascq as i32) == 27 {
break '__slate_break_1;
}
// sanitize in progress
if (sshdr.asc as i32) == 4 && (sshdr.ascq as i32) == 36 {
break '__slate_break_1;
}
// depopulation in progress
if (sshdr.asc as i32) == 4 && (sshdr.ascq as i32) == 37 {
break '__slate_break_1;
}
// depopulation restoration in progress
// Issue command to spin up drive when not ready
if spintime == 0 {
// Return immediately and start spin cycle
let mut start_cmd: [u8; 10] = {
let mut __t6: [u8; 10] = unsafe { core::mem::zeroed() };
__t6[0] = 27u8;
__t6[1] = 1u8;
__t6[4] = (if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_start_stop_pwr_cond() }) as i32) != 0 { 17i32 } else { 1i32 }) as u8;
__t6
};
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Spinning up disk...".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Spinning up disk...".as_ptr() as *const u8) };
}
unsafe { scsi_execute_cmd((*sdkp).device, start_cmd.as_ptr() as *const u8, 34u32, core::ptr::null_mut::<()>(), 0u32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
spintime_expire = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_add(100000u64);
spintime = 1;
}
// Wait 1 second for next try
unsafe { msleep(1000u32) };
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x01c.".as_ptr() as *const u8) };
// Wait for USB flash devices with slow firmware.
// Yes, this sense key/ASC combination shouldn't
// occur here.  It's characteristic of these devices.
} else {
if sense_valid != 0 && (sshdr.sense_key as i32) == 6 && (sshdr.asc as i32) == 40 {
if spintime == 0 {
spintime_expire = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_add(5000u64);
spintime = 1;
}
// Wait 1 second for next try
unsafe { msleep(1000u32) };
} else {
// we don't understand the sense code, so it's
// probably pointless to loop
if spintime == 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Unit Not Ready\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Unit Not Ready\n".as_ptr() as *const u8) };
}
sd_print_sense_hdr(sdkp, core::ptr::addr_of_mut!(sshdr));
}
break '__slate_break_1;
}
}
let __v4: bool = spintime != 0 && time_before_eq!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, spintime_expire);
if !__v4 {
break;
}
}
if spintime != 0 {
if scsi_status_is_good(the_result) {
let mut __v12: i32;
while false {
}
__v12 = unsafe { _printk(c"\x01cready\n".as_ptr() as *const u8) };
} else {
let mut __v14: i32;
while false {
}
__v14 = unsafe { _printk(c"\x01cnot responding...\n".as_ptr() as *const u8) };
}
}
}

static cmd__2490: [u8; 10] = {
let mut __t0: [u8; 10] = unsafe { core::mem::zeroed() };
__t0[0] = 0u8;
__t0
};

/// Determine whether disk supports Data Integrity Field.
fn sd_read_protection_type(sdkp: *mut scsi_disk, buffer: *mut u8) -> i32 {
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut r#type: u8 = 0;
if scsi_device_protection(sdp) == 0 || ((unsafe { *buffer.wrapping_offset(12 as isize) }) as i32) & 1 == 0 {
unsafe {
(*sdkp).protection_type = 0;
}
return 0;
}
r#type = (((unsafe { *buffer.wrapping_offset(12 as isize) }) as i32) >> 1i32 & 7).wrapping_add(1) as u8; // P_TYPE 0 = Type 1
if (r#type as i32) > 3 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"formatted with unsupported protection type %u. Disabling disk!\n".as_ptr() as *const u8, r#type as i32) };
} else {
unsafe { sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"formatted with unsupported protection type %u. Disabling disk!\n".as_ptr() as *const u8, r#type as i32) };
}
unsafe {
(*sdkp).protection_type = 0;
}
return 19i32.wrapping_neg();
}
unsafe {
(*sdkp).protection_type = r#type;
}
0
}

fn sd_config_protection(sdkp: *mut scsi_disk, lim: *mut queue_limits) {
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
if IS_ENABLED!() != 0 {
unsafe { sd_dif_config_host(sdkp, lim) };
}
if (unsafe { (*sdkp).protection_type }) == 0 {
return;
}
if scsi_host_dif_capable(unsafe { (*sdp).host }, (unsafe { (*sdkp).protection_type }) as u32) == 0 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Disabling DIF Type %u protection\n".as_ptr() as *const u8, (*sdkp).protection_type as i32) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Disabling DIF Type %u protection\n".as_ptr() as *const u8, (*sdkp).protection_type as i32) };
}
}
if true {
break;
}
}
unsafe {
(*sdkp).protection_type = 0;
}
}
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Enabling DIF Type %u protection\n".as_ptr() as *const u8, (*sdkp).protection_type as i32) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Enabling DIF Type %u protection\n".as_ptr() as *const u8, (*sdkp).protection_type as i32) };
}
}
if true {
break;
}
}
}

fn read_capacity_error(sdkp: *mut scsi_disk, sdp: *mut scsi_device, sshdr: *mut scsi_sense_hdr, sense_valid: i32, the_result: i32) {
if sense_valid != 0 {
sd_print_sense_hdr(sdkp, sshdr);
} else {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Sense not available.\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Sense not available.\n".as_ptr() as *const u8) };
}
}
// Set dirty bit for removable devices if not ready -
// sometimes drives will not report this properly.
if ((unsafe { (*sdp).__slate_bits_0.__get_removable() }) as i32) != 0 && sense_valid != 0 && ((unsafe { (*sshdr).sense_key }) as i32) == 2 {
set_media_not_present(sdkp);
}
// We used to set media_present to 0 here to indicate no media
// in the drive, but some drives fail read capacity even with
// media present, so we can't do that.
unsafe {
(*sdkp).capacity = 0;
}
// unknown mapped to zero - as usual
}

fn read_capacity_16(sdkp: *mut scsi_disk, sdp: *mut scsi_device, lim: *mut queue_limits, buffer: *mut u8) -> i32 {
let mut cmd: __SlateAlign16<[u8; 16]> = __SlateAlign16([0u8; 16]);
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let exec_args: scsi_exec_args = {
let mut __t0: scsi_exec_args = unsafe { core::mem::zeroed() };
__t0.sshdr = core::ptr::addr_of_mut!(sshdr);
__t0
};
let mut sense_valid: i32 = 0;
let mut the_result: i32 = 0;
let mut retries: i32 = 3;
let mut reset_retries: i32 = 10;
let mut alignment: u32 = 0;
let mut lba: u64 = 0;
let mut sector_size: u32 = 0;
if ((unsafe { (*sdp).__slate_bits_0.__get_no_read_capacity_16() }) as i32) != 0 {
return 22i32.wrapping_neg();
}
loop {
'__slate_continue_2: {
unsafe { memset(cmd.0.as_mut_ptr() as *mut (), 0i32, 16u64) };
cmd.0[0] = 158;
cmd.0[1] = 16;
cmd.0[13] = 32;
unsafe { memset(buffer as *mut (), 0i32, 32u64) };
the_result = unsafe { scsi_execute_cmd(sdp, cmd.0.as_mut_ptr() as *const u8, 34u32, buffer as *mut (), 32u32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
if the_result > 0 {
if media_not_present(sdkp, core::ptr::addr_of_mut!(sshdr)) != 0 {
return 19i32.wrapping_neg();
}
sense_valid = scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr) as i32;
if sense_valid != 0 && (sshdr.sense_key as i32) == 5 && ((sshdr.asc as i32) == 32 || (sshdr.asc as i32) == 36) && (sshdr.ascq as i32) == 0 {
return 22i32.wrapping_neg();
}
// Invalid Command Operation Code or
// Invalid Field in CDB, just retry
// silently with RC10
if sense_valid != 0 && (sshdr.sense_key as i32) == 6 && (sshdr.asc as i32) == 41 && (sshdr.ascq as i32) == 0 {
let __v1: i32 = reset_retries.wrapping_sub(1);
reset_retries = __v1;
if __v1 > 0 {
break '__slate_continue_2;
}
}
// Device reset might occur several times,
// give it one more chance
}
retries = retries.wrapping_sub(1);
}
if !(the_result != 0 && retries != 0) {
break;
}
}
if the_result != 0 {
sd_print_result(sdkp as *const scsi_disk, c"Read Capacity(16) failed".as_ptr() as *const u8, the_result);
read_capacity_error(sdkp, sdp, core::ptr::addr_of_mut!(sshdr), sense_valid, the_result);
return 22i32.wrapping_neg();
}
sector_size = get_unaligned_be32(buffer.wrapping_offset(8 as isize) as *const ());
lba = get_unaligned_be64(buffer.wrapping_offset(0 as isize) as *const ());
if sd_read_protection_type(sdkp, buffer) < 0 {
unsafe {
(*sdkp).capacity = 0;
}
return 19i32.wrapping_neg();
}
// Logical blocks per physical block exponent
unsafe {
(*sdkp).physical_block_size = ((1i32 << ((*buffer.wrapping_offset(13 as isize) as i32) & 15)) as u32).wrapping_mul(sector_size);
}
// RC basis
unsafe {
let __t1 = ((*buffer.wrapping_offset(12 as isize) as i32) >> 4i32 & 3) as u32;
(*sdkp).__slate_bits_0.__set_rc_basis(__t1);
}
// Lowest aligned logical block
alignment = (((((unsafe { *buffer.wrapping_offset(14 as isize) }) as i32) & 63) << 8i32 | ((unsafe { *buffer.wrapping_offset(15 as isize) }) as i32)) as u32).wrapping_mul(sector_size);
unsafe {
(*lim).alignment_offset = alignment;
}
if alignment != 0 && ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"physical block alignment offset: %u\n".as_ptr() as *const u8, alignment) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"physical block alignment offset: %u\n".as_ptr() as *const u8, alignment) };
}
}
if ((unsafe { *buffer.wrapping_offset(14 as isize) }) as i32) & 128 != 0 {
// LBPME
unsafe {
(*sdkp).__slate_bits_0.__set_lbpme(1u32);
}
if ((unsafe { *buffer.wrapping_offset(14 as isize) }) as i32) & 64 != 0 {
unsafe {
(*sdkp).__slate_bits_0.__set_lbprz(1u32);
}
}
// LBPRZ
}
unsafe {
(*sdkp).capacity = lba.wrapping_add(1);
}
sector_size as i32
}

fn read_capacity_10(sdkp: *mut scsi_disk, sdp: *mut scsi_device, buffer: *mut u8) -> i32 {
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut failure_defs: __SlateAlign16<[scsi_failure; 5]> = __SlateAlign16([{
let mut __t0: scsi_failure = unsafe { core::mem::zeroed() };
__t0.result = 2i32;
__t0.sense = 6u8;
__t0.asc = 58u8;
__t0
}, {
let mut __t1: scsi_failure = unsafe { core::mem::zeroed() };
__t1.result = 2i32;
__t1.sense = 2u8;
__t1.asc = 58u8;
__t1
}, {
let mut __t2: scsi_failure = unsafe { core::mem::zeroed() };
__t2.result = 2i32;
__t2.sense = 6u8;
__t2.asc = 41u8;
__t2.allowed = 10i8;
__t2
}, {
let mut __t3: scsi_failure = unsafe { core::mem::zeroed() };
__t3.result = 2147483647i32;
__t3.allowed = 3i8;
__t3
}, {
let mut __t4: scsi_failure = unsafe { core::mem::zeroed() };
__t4
}]);
// Do not retry Medium Not Present
// Device reset might occur several times so retry a lot
// Any other error not listed above retry 3 times
let mut failures: scsi_failures = {
let mut __t5: scsi_failures = unsafe { core::mem::zeroed() };
__t5.failure_definitions = failure_defs.0.as_mut_ptr() as *mut scsi_failure;
__t5
};
let exec_args: scsi_exec_args = {
let mut __t6: scsi_exec_args = unsafe { core::mem::zeroed() };
__t6.sshdr = core::ptr::addr_of_mut!(sshdr);
__t6.failures = core::ptr::addr_of_mut!(failures);
__t6
};
let mut sense_valid: i32 = 0;
let mut the_result: i32 = 0;
let mut lba: u64 = 0;
let mut sector_size: u32 = 0;
unsafe { memset(buffer as *mut (), 0i32, 8u64) };
the_result = unsafe { scsi_execute_cmd(sdp, core::ptr::addr_of!(cmd__2815) as *const u8, 34u32, buffer as *mut (), 8u32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
if the_result > 0 {
sense_valid = scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr) as i32;
if media_not_present(sdkp, core::ptr::addr_of_mut!(sshdr)) != 0 {
return 19i32.wrapping_neg();
}
}
if the_result != 0 {
sd_print_result(sdkp as *const scsi_disk, c"Read Capacity(10) failed".as_ptr() as *const u8, the_result);
read_capacity_error(sdkp, sdp, core::ptr::addr_of_mut!(sshdr), sense_valid, the_result);
return 22i32.wrapping_neg();
}
sector_size = get_unaligned_be32(buffer.wrapping_offset(4 as isize) as *const ());
lba = get_unaligned_be32(buffer.wrapping_offset(0 as isize) as *const ()) as u64;
if ((unsafe { (*sdp).__slate_bits_0.__get_no_read_capacity_16() }) as i32) != 0 && lba == 4294967295 {
// Some buggy (usb cardreader) devices return an lba of
// 0xffffffff when the want to report a size of 0 (with
// which they really mean no media is present)
unsafe {
(*sdkp).capacity = 0;
}
unsafe {
(*sdkp).physical_block_size = sector_size;
}
return sector_size as i32;
}
unsafe {
(*sdkp).capacity = lba.wrapping_add(1);
}
unsafe {
(*sdkp).physical_block_size = sector_size;
}
sector_size as i32
}

static cmd__2815: [u8; 10] = {
let mut __t0: [u8; 10] = unsafe { core::mem::zeroed() };
__t0[0] = 37u8;
__t0
};

fn sd_try_rc16_first(sdp: *mut scsi_device) -> i32 {
if ((unsafe { (*(*sdp).host).max_cmd_len }) as i32) < 16 {
return 0;
}
if ((unsafe { (*sdp).__slate_bits_0.__get_try_rc_10_first() }) as i32) != 0 {
return 0;
}
if ((unsafe { (*sdp).scsi_level }) as i32) > 5 {
return 1;
}
if scsi_device_protection(sdp) != 0 {
return 1;
}
0
}

/// read disk capacity
fn sd_read_capacity(sdkp: *mut scsi_disk, lim: *mut queue_limits, buffer: *mut u8) {
let mut old_sector_size: i32 = unsafe { core::mem::zeroed() };
let mut sdp: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut sector_size: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(sector_size) as *mut i32;
unsafe {
'__join_11: {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
sdp = (*sdkp).device;
if sd_try_rc16_first(sdp) != 0 {
*__slate_slot_5 = read_capacity_16(sdkp, sdp, lim, buffer);
if *__slate_slot_5 == 75i32.wrapping_neg() {
break '__join_11;
} else {
if *__slate_slot_5 == 19i32.wrapping_neg() {
return;
} else {
if *__slate_slot_5 < 0 {
*__slate_slot_5 = read_capacity_10(sdkp, sdp, buffer);
}
if *__slate_slot_5 < 0 {
return;
}
}
}
} else {
*__slate_slot_5 = read_capacity_10(sdkp, sdp, buffer);
if *__slate_slot_5 == 75i32.wrapping_neg() {
break '__join_11;
} else {
if *__slate_slot_5 < 0 {
return;
} else {
if (core::mem::size_of::<u64>() as u64) > 4 && (*sdkp).capacity > 4294967295 {
old_sector_size = *__slate_slot_5;
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Very big device. Trying to use READ CAPACITY(16).\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Very big device. Trying to use READ CAPACITY(16).\n".as_ptr() as *const u8);
}
*__slate_slot_5 = read_capacity_16(sdkp, sdp, lim, buffer);
if *__slate_slot_5 < 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Using 0xffffffff as device size\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Using 0xffffffff as device size\n".as_ptr() as *const u8);
}
(*sdkp).capacity = 4294967296u64;
*__slate_slot_5 = old_sector_size;
break '__join_11;
} else {
// Remember that READ CAPACITY(16) succeeded
(*sdp).__slate_bits_0.__set_try_rc_10_first(0u32);
}
}
}
}
}
// Some devices are known to return the total number of blocks,
// not the highest block number.  Some devices have versions
// which do this and others which do not.  Some devices we might
// suspect of doing this but we don't know for certain.
//
// If we know the reported capacity is wrong, decrement it.  If
// we can only guess, then assume the number of blocks is even
// (usually true but not always) and err on the side of lowering
// the capacity.
if ((*sdp).__slate_bits_0.__get_fix_capacity() as i32) != 0 || ((*sdp).__slate_bits_0.__get_guess_capacity() as i32) != 0 && (*sdkp).capacity & 1 != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Adjusting the sector count from its reported value: %llu\n".as_ptr() as *const u8, (*sdkp).capacity);
} else {
sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Adjusting the sector count from its reported value: %llu\n".as_ptr() as *const u8, (*sdkp).capacity);
}
let __v1: u64 = (*sdkp).capacity;
(*sdkp).capacity = __v1.wrapping_sub(1);
}
}
if *__slate_slot_5 == 0 {
*__slate_slot_5 = 512;
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Sector size 0 reported, assuming 512.\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Sector size 0 reported, assuming 512.\n".as_ptr() as *const u8);
}
}
if blk_validate_block_size(*__slate_slot_5 as u64) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Unsupported sector size %d.\n".as_ptr() as *const u8, *__slate_slot_5);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Unsupported sector size %d.\n".as_ptr() as *const u8, *__slate_slot_5);
}
// The user might want to re-format the drive with
// a supported sectorsize.  Once this happens, it
// would be relatively trivial to set the thing up.
// For this reason, we leave the thing in the table.
(*sdkp).capacity = 0;
// set a bogus sector size so the normal read/write
// logic in the block layer will eventually refuse any
// request on this device without tripping over power
// of two sector size assumptions
*__slate_slot_5 = 512;
}
(*lim).logical_block_size = *__slate_slot_5 as u32;
(*lim).physical_block_size = (*sdkp).physical_block_size;
(*(*sdkp).device).sector_size = *__slate_slot_5 as u32;
if (*sdkp).capacity > 4294967295 {
(*sdp).__slate_bits_0.__set_use_16_for_rw(1u32);
}
}
}

/// Print disk capacity
fn sd_print_capacity(sdkp: *mut scsi_disk, old_capacity: u64) {
let sector_size: i32 = (unsafe { (*(*sdkp).device).sector_size }) as i32;
let mut cap_str_2: [u8; 10] = [0u8; 10];
let mut cap_str_10: [u8; 10] = [0u8; 10];
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) == 0 && old_capacity == unsafe { (*sdkp).capacity } {
return;
}
unsafe { string_get_size((*sdkp).capacity, sector_size as u64, 1u32, cap_str_2.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 10]>() as i32) };
unsafe { string_get_size((*sdkp).capacity, sector_size as u64, 0u32, cap_str_10.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 10]>() as i32) };
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"%llu %d-byte logical blocks: (%s/%s)\n".as_ptr() as *const u8, (*sdkp).capacity, sector_size, cap_str_10.as_mut_ptr() as *mut u8, cap_str_2.as_mut_ptr() as *mut u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"%llu %d-byte logical blocks: (%s/%s)\n".as_ptr() as *const u8, (*sdkp).capacity, sector_size, cap_str_10.as_mut_ptr() as *mut u8, cap_str_2.as_mut_ptr() as *mut u8) };
}
if (unsafe { (*sdkp).physical_block_size }) != (sector_size as u32) {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"%u-byte physical blocks\n".as_ptr() as *const u8, (*sdkp).physical_block_size) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"%u-byte physical blocks\n".as_ptr() as *const u8, (*sdkp).physical_block_size) };
}
}
}

/// called with buffer of length 512
fn sd_do_mode_sense(sdkp: *mut scsi_disk, dbd: i32, modepage: i32, buffer: *mut u8, mut len: i32, data: *mut scsi_mode_data, sshdr: *mut scsi_sense_hdr) -> i32 {
// If we must use MODE SENSE(10), make sure that the buffer length
// is at least 8 bytes so that the mode sense header fits.
if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_use_10_for_ms() }) as i32) != 0 && len < 8 {
len = 8;
}
unsafe { scsi_mode_sense((*sdkp).device, dbd, modepage, 0i32, buffer, len, 30000i32, (*sdkp).max_retries, data, sshdr) }
}

/// read write protect setting, if possible - called only in sd_revalidate_disk()
/// called with buffer of length SD_BUF_SIZE
fn sd_read_write_protect_flag(sdkp: *mut scsi_disk, buffer: *mut u8) {
let mut res: i32 = 0;
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut data: scsi_mode_data = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(data) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let old_wp: i32 = (unsafe { (*sdkp).write_prot }) as i32;
unsafe { set_disk_ro((*sdkp).disk, false) };
if ((unsafe { (*sdp).__slate_bits_0.__get_skip_ms_page_3f() }) as i32) != 0 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Assuming Write Enabled\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Assuming Write Enabled\n".as_ptr() as *const u8) };
}
}
if true {
break;
}
}
return;
}
if ((unsafe { (*sdp).__slate_bits_0.__get_use_192_bytes_for_3f() }) as i32) != 0 {
res = sd_do_mode_sense(sdkp, 0i32, 63i32, buffer, 192i32, core::ptr::addr_of_mut!(data), core::ptr::null_mut::<scsi_sense_hdr>());
} else {
// First attempt: ask for all pages (0x3F), but only 4 bytes.
// We have to start carefully: some devices hang if we ask
// for more than is available.
res = sd_do_mode_sense(sdkp, 0i32, 63i32, buffer, 4i32, core::ptr::addr_of_mut!(data), core::ptr::null_mut::<scsi_sense_hdr>());
// Second attempt: ask for page 0 When only page 0 is
// implemented, a request for page 3F may return Sense Key
// 5: Illegal Request, Sense Code 24: Invalid field in
// CDB.
if res < 0 {
res = sd_do_mode_sense(sdkp, 0i32, 0i32, buffer, 4i32, core::ptr::addr_of_mut!(data), core::ptr::null_mut::<scsi_sense_hdr>());
}
// Third attempt: ask 255 bytes, as we did earlier.
if res < 0 {
res = sd_do_mode_sense(sdkp, 0i32, 63i32, buffer, 255i32, core::ptr::addr_of_mut!(data), core::ptr::null_mut::<scsi_sense_hdr>());
}
}
if res < 0 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Test WP failed, assume Write Enabled\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Test WP failed, assume Write Enabled\n".as_ptr() as *const u8) };
}
}
if true {
break;
}
}
} else {
unsafe {
(*sdkp).write_prot = ((data.device_specific as i32) & 128 != 0) as u8;
}
unsafe { set_disk_ro((*sdkp).disk, (*sdkp).write_prot != 0) };
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 || old_wp != ((unsafe { (*sdkp).write_prot }) as i32) {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Write Protect is %s\n".as_ptr() as *const u8, if (*sdkp).write_prot != 0 { c"on".as_ptr() as *mut u8 } else { c"off".as_ptr() as *mut u8 }) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Write Protect is %s\n".as_ptr() as *const u8, if (*sdkp).write_prot != 0 { c"on".as_ptr() as *mut u8 } else { c"off".as_ptr() as *mut u8 }) };
}
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x017".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Mode Sense: %4ph\n".as_ptr() as *const u8, buffer) };
} else {
unsafe { sdev_prefix_printk(c"\x017".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Mode Sense: %4ph\n".as_ptr() as *const u8, buffer) };
}
}
}
}

/// sd_read_cache_type - called only from sd_revalidate_disk()
/// called with buffer of length SD_BUF_SIZE
fn sd_read_cache_type(sdkp: *mut scsi_disk, buffer: *mut u8) {
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut spf: u8 = unsafe { core::mem::zeroed() };
let mut page_code: u8 = unsafe { core::mem::zeroed() };
let mut offset: i32 = unsafe { core::mem::zeroed() };
let mut old_dpofua: i32 = unsafe { core::mem::zeroed() };
let mut old_rcd: i32 = unsafe { core::mem::zeroed() };
let mut old_wce: i32 = unsafe { core::mem::zeroed() };
let mut sshdr: core::mem::MaybeUninit<scsi_sense_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut scsi_sense_hdr = core::ptr::addr_of_mut!(sshdr) as *mut scsi_sense_hdr;
let mut data: core::mem::MaybeUninit<scsi_mode_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut scsi_mode_data = core::ptr::addr_of_mut!(data) as *mut scsi_mode_data;
let mut first_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(first_len) as *mut i32;
let mut modepage: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut i32 = core::ptr::addr_of_mut!(modepage) as *mut i32;
let mut dbd: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut i32 = core::ptr::addr_of_mut!(dbd) as *mut i32;
let mut sdp: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut res: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(res) as *mut i32;
let mut len: i32 = unsafe { core::mem::zeroed() };
unsafe {
len = 0;
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
sdp = (*sdkp).device;
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 12usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
old_wce = (*sdkp).__slate_bits_0.__get_WCE() as i32;
old_rcd = (*sdkp).__slate_bits_0.__get_RCD() as i32;
old_dpofua = (*sdkp).__slate_bits_0.__get_DPOFUA() as i32;
if ((*sdkp).__slate_bits_0.__get_cache_override() as i32) != 0 {
return;
} else {
'__join_13: {
*__slate_slot_13 = 4;
if ((*sdp).__slate_bits_0.__get_skip_ms_page_8() as i32) != 0 {
if ((*sdp).r#type as i32) == 14 {
break '__join_13;
} else {
if ((*sdp).__slate_bits_0.__get_skip_ms_page_3f() as i32) != 0 {
break '__join_13;
} else {
*__slate_slot_14 = 63;
if ((*sdp).__slate_bits_0.__get_use_192_bytes_for_3f() as i32) != 0 {
*__slate_slot_13 = 192;
}
*__slate_slot_15 = 0;
}
}
} else {
if ((*sdp).r#type as i32) == 14 {
*__slate_slot_14 = 6;
*__slate_slot_15 = 8;
} else {
*__slate_slot_14 = 8;
*__slate_slot_15 = 0;
}
}
// cautiously ask
*__slate_slot_17 = sd_do_mode_sense(sdkp, *__slate_slot_15, *__slate_slot_14, buffer, *__slate_slot_13, core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11));
if *__slate_slot_17 >= 0 {
if (*__slate_slot_12).header_length == 0 {
*__slate_slot_14 = 6;
*__slate_slot_13 = 0;
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Missing header in MODE_SENSE response\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Missing header in MODE_SENSE response\n".as_ptr() as *const u8);
}
}
if true {
break;
}
}
}
// that went OK, now ask for the proper length
len = (*__slate_slot_12).length as i32;
// We're only interested in the first three bytes, actually.
// But the data cache page is defined for the first 20.
if len >= 3 {
if len > 512 {
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Truncating mode parameter data from %d to %d bytes\n".as_ptr() as *const u8, len, 512i32);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Truncating mode parameter data from %d to %d bytes\n".as_ptr() as *const u8, len, 512i32);
}
}
if true {
break;
}
}
len = 512;
}
if *__slate_slot_14 == 63 && ((*sdp).__slate_bits_0.__get_use_192_bytes_for_3f() as i32) != 0 {
len = 192;
}
// Get the data
if len > *__slate_slot_13 {
*__slate_slot_17 = sd_do_mode_sense(sdkp, *__slate_slot_15, *__slate_slot_14, buffer, len, core::ptr::addr_of_mut!(*__slate_slot_12), core::ptr::addr_of_mut!(*__slate_slot_11));
}
if *__slate_slot_17 == 0 {
offset = ((*__slate_slot_12).header_length as i32).wrapping_add((*__slate_slot_12).block_descriptor_length as i32);
'__join_64: {
'__join_71: {
while offset < len {
page_code = ((*buffer.wrapping_offset(offset as isize) as i32) & 63) as u8;
spf = ((*buffer.wrapping_offset(offset as isize) as i32) & 64) as u8;
if (page_code as i32) == 8 || (page_code as i32) == 6 {
break '__join_64;
} else {
// Go to the next page
if spf != 0 && len.wrapping_sub(offset) > 3 {
offset = offset.wrapping_add(4i32.wrapping_add((*buffer.wrapping_offset(offset.wrapping_add(2) as isize) as i32) << 8i32).wrapping_add(*buffer.wrapping_offset(offset.wrapping_add(3) as isize) as i32));
} else {
if spf == 0 && len.wrapping_sub(offset) > 1 {
offset = offset.wrapping_add(2i32.wrapping_add(*buffer.wrapping_offset(offset.wrapping_add(1) as isize) as i32));
} else {
break '__join_71;
}
}
}
}
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"No Caching mode page found\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"No Caching mode page found\n".as_ptr() as *const u8);
}
}
if true {
break '__join_13;
}
}
}
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Incomplete mode parameter data\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Incomplete mode parameter data\n".as_ptr() as *const u8);
}
}
if true {
break '__join_13;
}
}
}
// We're interested only in the first 3 bytes.
if len.wrapping_sub(offset) <= 2 {
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Incomplete mode parameter data\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Incomplete mode parameter data\n".as_ptr() as *const u8);
}
}
if true {
break '__join_13;
}
}
} else {
*__slate_slot_14 = page_code as i32;
if *__slate_slot_14 == 8 {
let __t1 = ((*buffer.wrapping_offset(offset.wrapping_add(2) as isize) as i32) & 4 != 0) as u32;
(*sdkp).__slate_bits_0.__set_WCE(__t1);
let __t2 = ((*buffer.wrapping_offset(offset.wrapping_add(2) as isize) as i32) & 1 != 0) as u32;
(*sdkp).__slate_bits_0.__set_RCD(__t2);
} else {
let __t3 = ((*buffer.wrapping_offset(offset.wrapping_add(2) as isize) as i32) & 1 == 0) as u32;
(*sdkp).__slate_bits_0.__set_WCE(__t3);
(*sdkp).__slate_bits_0.__set_RCD(0u32);
}
let __t0 = (((*__slate_slot_12).device_specific as i32) & 16 != 0) as u32;
(*sdkp).__slate_bits_0.__set_DPOFUA(__t0);
if ((*sdp).__slate_bits_0.__get_broken_fua() as i32) != 0 {
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Disabling FUA\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Disabling FUA\n".as_ptr() as *const u8);
}
}
if true {
break;
}
}
(*sdkp).__slate_bits_0.__set_DPOFUA(0u32);
} else {
if ((*sdkp).__slate_bits_0.__get_DPOFUA() as i32) != 0 && ((*(*sdkp).device).__slate_bits_0.__get_use_10_for_rw() as i32) == 0 && ((*(*sdkp).device).__slate_bits_0.__get_use_16_for_rw() as i32) == 0 {
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Uses READ/WRITE(6), disabling FUA\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Uses READ/WRITE(6), disabling FUA\n".as_ptr() as *const u8);
}
}
if true {
break;
}
}
(*sdkp).__slate_bits_0.__set_DPOFUA(0u32);
}
}
// No cache flush allowed for write protected devices
if ((*sdkp).__slate_bits_0.__get_WCE() as i32) != 0 && (*sdkp).write_prot != 0 {
(*sdkp).__slate_bits_0.__set_WCE(0u32);
}
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 || old_wce != ((*sdkp).__slate_bits_0.__get_WCE() as i32) || old_rcd != ((*sdkp).__slate_bits_0.__get_RCD() as i32) || old_dpofua != ((*sdkp).__slate_bits_0.__get_DPOFUA() as i32) {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Write cache: %s, read cache: %s, %s\n".as_ptr() as *const u8, if ((*sdkp).__slate_bits_0.__get_WCE() as i32) != 0 { c"enabled".as_ptr() as *mut u8 } else { c"disabled".as_ptr() as *mut u8 }, if ((*sdkp).__slate_bits_0.__get_RCD() as i32) != 0 { c"disabled".as_ptr() as *mut u8 } else { c"enabled".as_ptr() as *mut u8 }, if ((*sdkp).__slate_bits_0.__get_DPOFUA() as i32) != 0 { c"supports DPO and FUA".as_ptr() as *mut u8 } else { c"doesn't support DPO or FUA".as_ptr() as *mut u8 });
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Write cache: %s, read cache: %s, %s\n".as_ptr() as *const u8, if ((*sdkp).__slate_bits_0.__get_WCE() as i32) != 0 { c"enabled".as_ptr() as *mut u8 } else { c"disabled".as_ptr() as *mut u8 }, if ((*sdkp).__slate_bits_0.__get_RCD() as i32) != 0 { c"disabled".as_ptr() as *mut u8 } else { c"enabled".as_ptr() as *mut u8 }, if ((*sdkp).__slate_bits_0.__get_DPOFUA() as i32) != 0 { c"supports DPO and FUA".as_ptr() as *mut u8 } else { c"doesn't support DPO or FUA".as_ptr() as *mut u8 });
}
}
return;
}
}
}
}
if *__slate_slot_17 == 5i32.wrapping_neg() {
__v0 = scsi_sense_valid(core::ptr::addr_of_mut!(*__slate_slot_11) as *const scsi_sense_hdr);
} else {
__v0 = false;
}
if __v0 && ((*__slate_slot_11).sense_key as i32) == 5 && ((*__slate_slot_11).asc as i32) == 36 && ((*__slate_slot_11).ascq as i32) == 0 {
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Cache data unavailable\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Cache data unavailable\n".as_ptr() as *const u8);
}
}
if true {
break;
}
}
} else {
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Asking for cache data failed\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Asking for cache data failed\n".as_ptr() as *const u8);
}
}
if true {
break;
}
}
}
// Invalid field in CDB
}
if ((*sdp).__slate_bits_0.__get_wce_default_on() as i32) != 0 {
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Assuming drive cache: write back\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Assuming drive cache: write back\n".as_ptr() as *const u8);
}
}
if true {
break;
}
}
(*sdkp).__slate_bits_0.__set_WCE(1u32);
} else {
loop {
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Assuming drive cache: write through\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Assuming drive cache: write through\n".as_ptr() as *const u8);
}
}
if true {
break;
}
}
(*sdkp).__slate_bits_0.__set_WCE(0u32);
}
(*sdkp).__slate_bits_0.__set_RCD(0u32);
(*sdkp).__slate_bits_0.__set_DPOFUA(0u32);
}
}
}

fn sd_is_perm_stream(sdkp: *mut scsi_disk, stream_id: u32) -> bool {
let mut cdb: __SlateAlign16<[u8; 16]> = __SlateAlign16({
let mut __t0: [u8; 16] = unsafe { core::mem::zeroed() };
__t0[0] = 158u8;
__t0[1] = 22u8;
__t0
});
let mut buf: __SlateRecord_sd_c_3288 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(buf) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let exec_args: scsi_exec_args = {
let mut __t1: scsi_exec_args = unsafe { core::mem::zeroed() };
__t1.sshdr = core::ptr::addr_of_mut!(sshdr);
__t1
};
let mut res: i32 = 0;
put_unaligned_be16(stream_id as u16, (cdb.0.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) as *mut ());
put_unaligned_be32(core::mem::size_of::<__SlateRecord_sd_c_3288>() as u32, (cdb.0.as_mut_ptr() as *mut u8).wrapping_offset(10 as isize) as *mut ());
res = unsafe { scsi_execute_cmd(sdev, cdb.0.as_mut_ptr() as *const u8, 34u32, core::ptr::addr_of_mut!(buf) as *mut (), core::mem::size_of::<__SlateRecord_sd_c_3288>() as u32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
if res < 0 {
return false;
}
let __v0: bool = scsi_status_is_check_condition(res) != 0 && scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr);
if __v0 {
sd_print_sense_hdr(sdkp, core::ptr::addr_of_mut!(sshdr));
}
if res != 0 {
return false;
}
if (get_unaligned_be32(core::ptr::addr_of_mut!(buf.h.len) as *const ()) as u64) < (core::mem::size_of::<scsi_stream_status>() as u64) {
return false;
}
(buf.s.__slate_bits_0.__get_perm() as i32) != 0
}

fn sd_read_io_hints(sdkp: *mut scsi_disk, buffer: *mut u8) {
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut desc: *const scsi_io_group_descriptor = unsafe { core::mem::zeroed() };
let mut permanent_stream_count_old: u16 = 0;
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut data: scsi_mode_data = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(data) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let mut res: i32 = 0;
if (unsafe { (*sdp).sdev_bflags }) & 17179869184u64 != 0 {
return;
}
res = unsafe { scsi_mode_sense(sdp, 8i32, 10i32, 5i32, buffer, 512i32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of_mut!(data), core::ptr::addr_of_mut!(sshdr)) }; // dbd=
// modepage=
// subpage=
if res < 0 {
return;
}
let start: *const scsi_io_group_descriptor = ((unsafe { (((buffer as *const u8).wrapping_offset((data.header_length as i32) as isize) as *mut ()) as *const u8).wrapping_offset(16 as isize) }) as *mut ()) as *const scsi_io_group_descriptor;
let end: *const scsi_io_group_descriptor = ((unsafe { (buffer as *const u8).wrapping_offset(ALIGN_DOWN!(((data.header_length as i32) as u32).wrapping_add(data.length), core::mem::size_of::<scsi_io_group_descriptor>() as u64) as isize) }) as *mut ()) as *const scsi_io_group_descriptor;
// From "SBC-5 Constrained Streams with Data Lifetimes": Device severs
// should assign the lowest numbered stream identifiers to permanent
// streams.
desc = start;
while desc < end {
let __v0: bool = ((unsafe { (*desc).__slate_bits_0.__get_st_enble() }) as i32) == 0 || !sd_is_perm_stream(sdkp, ((unsafe { desc.offset_from(start as *const scsi_io_group_descriptor) }) as i64) as u32);
if __v0 {
break;
}
let __v2: *const scsi_io_group_descriptor = desc;
desc = __v2.wrapping_offset(1 as isize);
}
permanent_stream_count_old = unsafe { (*sdkp).permanent_stream_count };
unsafe {
(*sdkp).permanent_stream_count = (desc.offset_from(start as *const scsi_io_group_descriptor) as i64) as u16;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_rscs() }) as i32) != 0 && ((unsafe { (*sdkp).permanent_stream_count }) as i32) < 2 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Unexpected: RSCS has been set and the permanent stream count is %u\n".as_ptr() as *const u8, (*sdkp).permanent_stream_count as i32) };
} else {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Unexpected: RSCS has been set and the permanent stream count is %u\n".as_ptr() as *const u8, (*sdkp).permanent_stream_count as i32) };
}
} else {
if ((unsafe { (*sdkp).permanent_stream_count }) as i32) != (permanent_stream_count_old as i32) {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"permanent stream count = %d\n".as_ptr() as *const u8, (*sdkp).permanent_stream_count as i32) };
} else {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"permanent stream count = %d\n".as_ptr() as *const u8, (*sdkp).permanent_stream_count as i32) };
}
}
}
}

/// The ATO bit indicates whether the DIF application tag is available
/// for use by the operating system.
fn sd_read_app_tag_own(sdkp: *mut scsi_disk, buffer: *mut u8) {
let mut res: i32 = 0;
let mut offset: i32 = 0;
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut data: scsi_mode_data = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(data) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
if ((unsafe { (*sdp).r#type }) as i32) != 0 && ((unsafe { (*sdp).r#type }) as i32) != 20 {
return;
}
if ((unsafe { (*sdkp).protection_type }) as i32) == 0 {
return;
}
res = unsafe { scsi_mode_sense(sdp, 1i32, 10i32, 0i32, buffer, 36i32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of_mut!(data), core::ptr::addr_of_mut!(sshdr)) };
if res < 0 || data.header_length == 0 || data.length < 6 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"getting Control mode page failed, assume no ATO\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"getting Control mode page failed, assume no ATO\n".as_ptr() as *const u8) };
}
}
if true {
break;
}
}
let __v1: bool = res == 5i32.wrapping_neg() && scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr);
if __v1 {
sd_print_sense_hdr(sdkp, core::ptr::addr_of_mut!(sshdr));
}
return;
}
offset = (data.header_length as i32).wrapping_add(data.block_descriptor_length as i32);
if ((unsafe { *buffer.wrapping_offset(offset as isize) }) as i32) & 63 != 10 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"ATO Got wrong page\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"ATO Got wrong page\n".as_ptr() as *const u8) };
}
}
if true {
break;
}
}
return;
}
if ((unsafe { *buffer.wrapping_offset(offset.wrapping_add(5) as isize) }) as i32) & 128 == 0 {
return;
}
unsafe {
(*sdkp).__slate_bits_0.__set_ATO(1u32);
}
return;
}

fn sd_discard_mode(sdkp: *mut scsi_disk) -> u32 {
if ((unsafe { (*sdkp).__slate_bits_0.__get_lbpme() }) as i32) == 0 {
return 0;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_lbpvpd() }) as i32) == 0 {
// LBP VPD page not provided
if (unsafe { (*sdkp).max_unmap_blocks }) != 0 {
return 1;
}
return 2;
}
// LBP VPD page tells us what to use
if ((unsafe { (*sdkp).__slate_bits_0.__get_lbpu() }) as i32) != 0 && (unsafe { (*sdkp).max_unmap_blocks }) != 0 {
return 1;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_lbpws() }) as i32) != 0 {
return 2;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_lbpws10() }) as i32) != 0 {
return 3;
}
5
}

// Query disk device for preferred I/O sizes.
fn sd_read_block_limits(sdkp: *mut scsi_disk, lim: *mut queue_limits) {
let mut desc_count: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(desc_count) as *mut u32;
let mut lba_count: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u32 = core::ptr::addr_of_mut!(lba_count) as *mut u32;
let mut __v2: bool = unsafe { core::mem::zeroed() };
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_rcu_660: *mut scsi_vpd = unsafe { core::mem::zeroed() };
let mut vpd: core::mem::MaybeUninit<*mut scsi_vpd> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut scsi_vpd = core::ptr::addr_of_mut!(vpd) as *mut *mut scsi_vpd;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
rcu_read_lock();
while false {
}
let __v4: *mut scsi_vpd = core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*sdkp).device).vpd_pgb0) as *const *mut scsi_vpd);
__UNIQUE_ID_rcu_660 = __v4;
loop {
if false {
if false {
__v2 = true;
} else {
__v2 = rcu_read_lock_held() != 0;
}
__v3 = !__v2;
} else {
__v3 = false;
}
if !__v3 {
break;
}
}
*__slate_slot_7 = __UNIQUE_ID_rcu_660;
if !((*__slate_slot_7).is_null() || (*(*__slate_slot_7)).len < 16) {
(*sdkp).min_xfer_blocks = get_unaligned_be16((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(6 as isize) as *const ()) as u32;
(*sdkp).max_xfer_blocks = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(8 as isize) as *const ());
(*sdkp).opt_xfer_blocks = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(12 as isize) as *const ());
if (*(*__slate_slot_7)).len >= 64 {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
(*sdkp).max_ws_blocks = get_unaligned_be64((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(36 as isize) as *const ()) as u32;
if ((*sdkp).__slate_bits_0.__get_lbpme() as i32) != 0 {
*__slate_slot_1 = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(20 as isize) as *const ());
*__slate_slot_0 = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(24 as isize) as *const ());
if *__slate_slot_1 != 0 && *__slate_slot_0 != 0 {
(*sdkp).max_unmap_blocks = *__slate_slot_1;
}
(*sdkp).unmap_granularity = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(28 as isize) as *const ());
if (*(core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(32 as isize) as i32) & 128 != 0 {
(*sdkp).unmap_alignment = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(32 as isize) as *const ()) & 2147483647u32;
}
}
(*sdkp).max_atomic = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(44 as isize) as *const ());
(*sdkp).atomic_alignment = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(48 as isize) as *const ());
(*sdkp).atomic_granularity = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(52 as isize) as *const ());
(*sdkp).max_atomic_with_boundary = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(56 as isize) as *const ());
(*sdkp).max_atomic_boundary = get_unaligned_be32((core::ptr::addr_of_mut!((*(*__slate_slot_7)).data) as *mut u8).wrapping_offset(60 as isize) as *const ());
sd_config_atomic(sdkp, lim);
}
}
rcu_read_unlock();
}
}

// Parse the Block Limits Extension VPD page (0xb7)
fn sd_read_block_limits_ext(sdkp: *mut scsi_disk) {
let mut vpd: *mut scsi_vpd = unsafe { core::mem::zeroed() };
rcu_read_lock();
let mut __v0: *mut scsi_vpd;
let mut __v1: *mut scsi_vpd;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_663() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*sdkp).device).vpd_pgb7) as *const *mut scsi_vpd) };
let __UNIQUE_ID_rcu_662: *mut scsi_vpd = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = false || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_662;
vpd = __v0;
if !vpd.is_null() && (unsafe { (*vpd).len }) >= 6 {
unsafe {
let __t0 = ((*(core::ptr::addr_of_mut!((*vpd).data) as *mut u8).wrapping_offset(5 as isize) as i32) & 1) as u32;
(*sdkp).__slate_bits_0.__set_rscs(__t0);
}
}
rcu_read_unlock();
}

// Query block device characteristics
fn sd_read_block_characteristics(sdkp: *mut scsi_disk, lim: *mut queue_limits) {
let mut vpd: *mut scsi_vpd = unsafe { core::mem::zeroed() };
let mut rot: u16 = 0;
rcu_read_lock();
let mut __v0: *mut scsi_vpd;
let mut __v1: *mut scsi_vpd;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_665() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*sdkp).device).vpd_pgb1) as *const *mut scsi_vpd) };
let __UNIQUE_ID_rcu_664: *mut scsi_vpd = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = false || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_664;
vpd = __v0;
if vpd.is_null() || (unsafe { (*vpd).len }) <= 8 {
rcu_read_unlock();
return;
}
rot = get_unaligned_be16(unsafe { core::ptr::addr_of_mut!((*vpd).data) as *mut u8 }.wrapping_offset(4 as isize) as *const ());
unsafe {
let __t0 = ((*(core::ptr::addr_of_mut!((*vpd).data) as *mut u8).wrapping_offset(8 as isize) as i32) >> 4i32 & 3) as u32;
(*sdkp).__slate_bits_0.__set_zoned(__t0);
}
rcu_read_unlock();
if (rot as i32) == 1 {
let __v8: u32 = (unsafe { (*lim).features }) & 4294967283u32;
unsafe {
(*lim).features = __v8;
}
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) == 0 {
return;
}
if ((unsafe { (*(*sdkp).device).r#type }) as i32) == 20 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Host-managed zoned block device\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Host-managed zoned block device\n".as_ptr() as *const u8) };
}
} else {
if ((unsafe { (*sdkp).__slate_bits_0.__get_zoned() }) as i32) == 1 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Host-aware SMR disk used as regular disk\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Host-aware SMR disk used as regular disk\n".as_ptr() as *const u8) };
}
} else {
if ((unsafe { (*sdkp).__slate_bits_0.__get_zoned() }) as i32) == 2 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Drive-managed SMR disk\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Drive-managed SMR disk\n".as_ptr() as *const u8) };
}
}
}
}
}

// sd_read_block_provisioning - Query provisioning VPD page
// @sdkp: disk to query
fn sd_read_block_provisioning(sdkp: *mut scsi_disk) {
let mut vpd: *mut scsi_vpd = unsafe { core::mem::zeroed() };
if ((unsafe { (*sdkp).__slate_bits_0.__get_lbpme() }) as i32) == 0 {
return;
}
rcu_read_lock();
let mut __v0: *mut scsi_vpd;
let mut __v1: *mut scsi_vpd;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_667() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*(*sdkp).device).vpd_pgb2) as *const *mut scsi_vpd) };
let __UNIQUE_ID_rcu_666: *mut scsi_vpd = __v1;
loop {
let __v4: bool = if false {
let __v5: bool = false || rcu_read_lock_held() != 0;
!__v5
} else {
false
};
if !__v4 {
break;
}
}
__v0 = __UNIQUE_ID_rcu_666;
vpd = __v0;
if vpd.is_null() || (unsafe { (*vpd).len }) < 8 {
rcu_read_unlock();
return;
}
unsafe {
(*sdkp).__slate_bits_0.__set_lbpvpd(1u32);
}
unsafe {
let __t0 = ((*(core::ptr::addr_of_mut!((*vpd).data) as *mut u8).wrapping_offset(5 as isize) as i32) >> 7i32 & 1) as u32;
(*sdkp).__slate_bits_0.__set_lbpu(__t0);
}
// UNMAP
unsafe {
let __t1 = ((*(core::ptr::addr_of_mut!((*vpd).data) as *mut u8).wrapping_offset(5 as isize) as i32) >> 6i32 & 1) as u32;
(*sdkp).__slate_bits_0.__set_lbpws(__t1);
}
// WRITE SAME(16) w/ UNMAP
unsafe {
let __t2 = ((*(core::ptr::addr_of_mut!((*vpd).data) as *mut u8).wrapping_offset(5 as isize) as i32) >> 5i32 & 1) as u32;
(*sdkp).__slate_bits_0.__set_lbpws10(__t2);
}
// WRITE SAME(10) w/ UNMAP
rcu_read_unlock();
}

fn sd_read_write_same(sdkp: *mut scsi_disk, buffer: *mut u8) {
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
if ((unsafe { (*(*sdev).host).__slate_bits_0.__get_no_write_same() }) as i32) != 0 {
unsafe {
(*sdev).__slate_bits_0.__set_no_write_same(1u32);
}
return;
}
if (unsafe { scsi_report_opcode(sdev, buffer, 512u32, 18u8, 0u16) }) < 0 {
unsafe {
(*sdev).__slate_bits_0.__set_no_report_opcodes(1u32);
}
// Disable WRITE SAME if REPORT SUPPORTED OPERATION CODES is
// unsupported and this is an ATA device.
if ((unsafe { (*sdev).__slate_bits_0.__get_is_ata() }) as i32) != 0 {
unsafe {
(*sdev).__slate_bits_0.__set_no_write_same(1u32);
}
}
}
if (unsafe { scsi_report_opcode(sdev, buffer, 512u32, 147u8, 0u16) }) == 1 {
unsafe {
(*sdkp).__slate_bits_0.__set_ws16(1u32);
}
}
if (unsafe { scsi_report_opcode(sdev, buffer, 512u32, 65u8, 0u16) }) == 1 {
unsafe {
(*sdkp).__slate_bits_0.__set_ws10(1u32);
}
}
}

fn sd_read_security(sdkp: *mut scsi_disk, buffer: *mut u8) {
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
if ((unsafe { (*sdev).__slate_bits_0.__get_security_supported() }) as i32) == 0 {
return;
}
let __v0: bool = (unsafe { scsi_report_opcode(sdev, buffer, 512u32, 162u8, 0u16) }) == 1 && (unsafe { scsi_report_opcode(sdev, buffer, 512u32, 181u8, 0u16) }) == 1;
if __v0 {
unsafe {
(*sdkp).__slate_bits_0.__set_security(1u32);
}
}
}

fn sd64_to_sectors(sdkp: *mut scsi_disk, buf: *mut u8) -> u64 {
logical_to_sectors(unsafe { (*sdkp).device }, get_unaligned_be64(buf as *const ()))
}

/// sd_read_cpr - Query concurrent positioning ranges
/// @sdkp:	disk to query
fn sd_read_cpr(sdkp: *mut scsi_disk) {
let mut __slate_compound_16: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_18: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut desc: core::mem::MaybeUninit<*mut u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut *mut u8 = core::ptr::addr_of_mut!(desc) as *mut *mut u8;
let mut buf_len: i32 = unsafe { core::mem::zeroed() };
let mut vpd_len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(vpd_len) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut nr_cpr: u32 = unsafe { core::mem::zeroed() };
let mut buffer: *mut u8 = unsafe { core::mem::zeroed() };
let mut iars: *mut blk_independent_access_ranges = unsafe { core::mem::zeroed() };
unsafe {
iars = core::ptr::null_mut::<blk_independent_access_ranges>();
buffer = core::ptr::null_mut::<u8>();
nr_cpr = 0;
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
buf_len = 512;
let __slate_auto_bytes_4 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 8usize);
// We need to have the capacity set first for the block layer to be
// able to check the ranges.
if ((*sdkp).__slate_bits_0.__get_first_scan() as i32) != 0 {
return;
} else {
'__join_5: {
if (*sdkp).capacity != 0 {
// Concurrent Positioning Ranges VPD: there can be at most 256 ranges,
// leading to a maximum page size of 64 + 256*32 bytes.
buf_len = 8256i32;
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_5 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_6 = _kmalloc_noprof(buf_len as u64, 3264u32, *{
__slate_compound_16 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_16)
});
alloc_tag_restore!();
} else {
*__slate_slot_6 = _kmalloc_noprof(buf_len as u64, 3264u32, *{
__slate_compound_18 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_18)
});
}
buffer = *__slate_slot_6 as *mut u8;
if buffer.is_null() {
__v4 = true;
} else {
__v4 = scsi_get_vpd_page((*sdkp).device, 185u8, buffer, buf_len) != 0;
}
if !__v4 {
// We must have at least a 64B header and one 32B range descriptor
*__slate_slot_11 = (get_unaligned_be16(buffer.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4);
if *__slate_slot_11 > buf_len || *__slate_slot_11 < 96i32 || *__slate_slot_11 & 31 != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Invalid Concurrent Positioning Ranges VPD page\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Invalid Concurrent Positioning Ranges VPD page\n".as_ptr() as *const u8);
}
} else {
nr_cpr = ((*__slate_slot_11).wrapping_sub(64) / 32) as u32;
if nr_cpr == 1 {
nr_cpr = 0;
} else {
iars = disk_alloc_independent_access_ranges((*sdkp).disk, nr_cpr as i32);
if iars.is_null() {
nr_cpr = 0;
} else {
*__slate_slot_9 = buffer.wrapping_offset(64 as isize);
*__slate_slot_12 = 0;
loop {
if (*__slate_slot_12 as u32) < nr_cpr {
if (*(*__slate_slot_9).wrapping_offset(0 as isize) as i32) != *__slate_slot_12 {
break;
} else {
(*(core::ptr::addr_of_mut!((*iars).ia_range) as *mut blk_independent_access_range).wrapping_offset(*__slate_slot_12 as isize)).sector = sd64_to_sectors(sdkp, (*__slate_slot_9).wrapping_offset(8 as isize));
(*(core::ptr::addr_of_mut!((*iars).ia_range) as *mut blk_independent_access_range).wrapping_offset(*__slate_slot_12 as isize)).nr_sectors = sd64_to_sectors(sdkp, (*__slate_slot_9).wrapping_offset(16 as isize));
let __v3: i32 = *__slate_slot_12;
*__slate_slot_12 = __v3.wrapping_add(1);
let __v1: *mut u8 = *__slate_slot_9;
*__slate_slot_9 = __v1.wrapping_offset(32 as isize);
}
} else {
break '__join_5;
}
}
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Invalid Concurrent Positioning Range number\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x013".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Invalid Concurrent Positioning Range number\n".as_ptr() as *const u8);
}
nr_cpr = 0;
}
}
}
}
}
}
disk_set_independent_access_ranges((*sdkp).disk, iars);
if nr_cpr != 0 && (((*sdkp).nr_actuators as i32) as u32) != nr_cpr {
if !unsafe { (*sdkp).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"%u concurrent positioning ranges\n".as_ptr() as *const u8, nr_cpr);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"%u concurrent positioning ranges\n".as_ptr() as *const u8, nr_cpr);
}
(*sdkp).nr_actuators = nr_cpr as u8;
}
kfree(buffer as *const ());
}
}
}

fn sd_validate_min_xfer_size(sdkp: *mut scsi_disk) -> bool {
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let min_xfer_bytes: u32 = logical_to_bytes(sdp, (unsafe { (*sdkp).min_xfer_blocks }) as u64);
if (unsafe { (*sdkp).min_xfer_blocks }) == 0 {
return false;
}
if min_xfer_bytes & unsafe { (*sdkp).physical_block_size }.wrapping_sub(1) != 0 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Preferred minimum I/O size %u bytes not a multiple of physical block size (%u bytes)\n".as_ptr() as *const u8, min_xfer_bytes, (*sdkp).physical_block_size) };
} else {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Preferred minimum I/O size %u bytes not a multiple of physical block size (%u bytes)\n".as_ptr() as *const u8, min_xfer_bytes, (*sdkp).physical_block_size) };
}
}
if true {
break;
}
}
unsafe {
(*sdkp).min_xfer_blocks = 0;
}
return false;
}
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Preferred minimum I/O size %u bytes\n".as_ptr() as *const u8, min_xfer_bytes) };
} else {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Preferred minimum I/O size %u bytes\n".as_ptr() as *const u8, min_xfer_bytes) };
}
}
if true {
break;
}
}
true
}

/// Determine the device's preferred I/O size for reads and writes
/// unless the reported value is unreasonably small, large, not a
/// multiple of the physical block size, or simply garbage.
fn sd_validate_opt_xfer_size(sdkp: *mut scsi_disk, dev_max: u32) -> bool {
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let opt_xfer_bytes: u32 = logical_to_bytes(sdp, (unsafe { (*sdkp).opt_xfer_blocks }) as u64);
let min_xfer_bytes: u32 = logical_to_bytes(sdp, (unsafe { (*sdkp).min_xfer_blocks }) as u64);
if (unsafe { (*sdkp).opt_xfer_blocks }) == 0 {
return false;
}
if (unsafe { (*sdkp).opt_xfer_blocks }) > dev_max {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Optimal transfer size %u logical blocks > dev_max (%u logical blocks)\n".as_ptr() as *const u8, (*sdkp).opt_xfer_blocks, dev_max) };
} else {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Optimal transfer size %u logical blocks > dev_max (%u logical blocks)\n".as_ptr() as *const u8, (*sdkp).opt_xfer_blocks, dev_max) };
}
}
if true {
break;
}
}
return false;
}
if (unsafe { (*sdkp).opt_xfer_blocks }) > 65535 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Optimal transfer size %u logical blocks > sd driver limit (%u logical blocks)\n".as_ptr() as *const u8, (*sdkp).opt_xfer_blocks, 65535u32) };
} else {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Optimal transfer size %u logical blocks > sd driver limit (%u logical blocks)\n".as_ptr() as *const u8, (*sdkp).opt_xfer_blocks, 65535u32) };
}
}
if true {
break;
}
}
return false;
}
if (opt_xfer_bytes as u64) < _AC!() << 12i32 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Optimal transfer size %u bytes < PAGE_SIZE (%u bytes)\n".as_ptr() as *const u8, opt_xfer_bytes, (_AC!() << 12i32) as u32) };
} else {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Optimal transfer size %u bytes < PAGE_SIZE (%u bytes)\n".as_ptr() as *const u8, opt_xfer_bytes, (_AC!() << 12i32) as u32) };
}
}
if true {
break;
}
}
return false;
}
if min_xfer_bytes != 0 && opt_xfer_bytes % min_xfer_bytes != 0 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Optimal transfer size %u bytes not a multiple of preferred minimum block size (%u bytes)\n".as_ptr() as *const u8, opt_xfer_bytes, min_xfer_bytes) };
} else {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Optimal transfer size %u bytes not a multiple of preferred minimum block size (%u bytes)\n".as_ptr() as *const u8, opt_xfer_bytes, min_xfer_bytes) };
}
}
if true {
break;
}
}
return false;
}
if opt_xfer_bytes & unsafe { (*sdkp).physical_block_size }.wrapping_sub(1) != 0 {
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Optimal transfer size %u bytes not a multiple of physical block size (%u bytes)\n".as_ptr() as *const u8, opt_xfer_bytes, (*sdkp).physical_block_size) };
} else {
unsafe { sdev_prefix_printk(c"\x014".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Optimal transfer size %u bytes not a multiple of physical block size (%u bytes)\n".as_ptr() as *const u8, opt_xfer_bytes, (*sdkp).physical_block_size) };
}
}
if true {
break;
}
}
return false;
}
loop {
if ((unsafe { (*sdkp).__slate_bits_0.__get_first_scan() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Optimal transfer size %u bytes\n".as_ptr() as *const u8, opt_xfer_bytes) };
} else {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Optimal transfer size %u bytes\n".as_ptr() as *const u8, opt_xfer_bytes) };
}
}
if true {
break;
}
}
true
}

fn sd_read_block_zero(sdkp: *mut scsi_disk) {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let sdev: *mut scsi_device = unsafe { (*sdkp).device };
let buf_len: u32 = unsafe { (*sdev).sector_size };
let mut cmd: __SlateAlign16<[u8; 16]> = __SlateAlign16({
let mut __t0: [u8; 16] = unsafe { core::mem::zeroed() };
__t0
});
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(buf_len as u64, 3264u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(buf_len as u64, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
let buffer: *mut u8 = __v0 as *mut u8;
if buffer.is_null() {
return;
}
if ((unsafe { (*sdev).__slate_bits_0.__get_use_16_for_rw() }) as i32) != 0 {
cmd.0[0] = 136;
put_unaligned_be64(0u64, (cmd.0.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as *mut ()); // Logical block address 0
put_unaligned_be32(1u32, (cmd.0.as_mut_ptr() as *mut u8).wrapping_offset(10 as isize) as *mut ()); // Transfer 1 logical block
} else {
cmd.0[0] = 40;
put_unaligned_be32(0u32, (cmd.0.as_mut_ptr() as *mut u8).wrapping_offset(2 as isize) as *mut ()); // Logical block address 0
put_unaligned_be16(1u16, (cmd.0.as_mut_ptr() as *mut u8).wrapping_offset(7 as isize) as *mut ()); // Transfer 1 logical block
}
unsafe { scsi_execute_cmd((*sdkp).device, cmd.0.as_mut_ptr() as *const u8, 34u32, buffer as *mut (), buf_len, 30000i32, (*sdkp).max_retries, core::ptr::null::<scsi_exec_args>()) };
unsafe { kfree(buffer as *const ()) };
}

// sd_revalidate_disk - called the first time a new disk is seen,
// performs disk spin up, read_capacity, etc.
// @disk: struct gendisk we care about
/// sd_unlock_native_capacity - unlock native capacity
/// @disk: struct gendisk to set capacity for
///
/// Block layer calls this function if it detects that partitions
/// on @disk reach beyond the end of the device.  If the SCSI host
/// implements ->unlock_native_capacity() method, it's invoked to
/// give it a chance to adjust the device capacity.
///
/// CONTEXT:
/// Defined by block layer.  Might sleep.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_unlock_native_capacity")]
extern "C" fn sd_unlock_native_capacity(disk: *mut gendisk) {
let sdev: *mut scsi_device = unsafe { (*scsi_disk(disk)).device };
if (unsafe { (*(*(*sdev).host).hostt).unlock_native_capacity }) != None {
unsafe { unsafe { (*(*(*sdev).host).hostt).unlock_native_capacity }.unwrap()(sdev) };
}
}

static sd_fops: __SlateConst<block_device_operations> = __SlateConst({
let mut __t0: block_device_operations = unsafe { core::mem::zeroed() };
__t0.open = Some(sd_open);
__t0.release = Some(sd_release);
__t0.ioctl = Some(sd_ioctl);
__t0.compat_ioctl = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut block_device, u32, u32, u64) -> i32>>(blkdev_compat_ptr_ioctl as *const ()) };
__t0.check_events = Some(sd_check_events);
__t0.unlock_native_capacity = Some(sd_unlock_native_capacity);
__t0.getgeo = Some(sd_getgeo);
__t0.free_disk = Some(scsi_disk_free_disk);
__t0.report_zones = None;
__t0.get_unique_id = Some(sd_get_unique_id);
__t0.owner = core::ptr::null_mut::<module>();
__t0.pr_ops = unsafe { core::ptr::addr_of!(sd_pr_ops.0) };
__t0
});

/// sd_format_disk_name - format disk name
/// @prefix: name prefix - ie. "sd" for SCSI disks
/// @index: index of the disk to format name for
/// @buf: output buffer
/// @buflen: length of the output buffer
///
/// SCSI disk names starts at sda.  The 26th device is sdz and the
/// 27th is sdaa.  The last one for two lettered suffix is sdzz
/// which is followed by sdaaa.
///
/// This is basically 26 base counting with one extra 'nil' entry
/// at the beginning from the second digit on and can be
/// determined using similar method as 26 base conversion with the
/// index shifted -1 after each digit is computed.
///
/// CONTEXT:
/// Don't care.
///
/// RETURNS:
/// 0 on success, -errno on failure.
fn sd_format_disk_name(prefix: *mut u8, mut index: i32, buf: *mut u8, buflen: i32) -> i32 {
let base: i32 = 26i32;
let begin: *mut u8 = buf.wrapping_offset((unsafe { strlen(prefix as *const u8) }) as isize);
let end: *mut u8 = buf.wrapping_offset(buflen as isize);
let mut p: *mut u8 = unsafe { core::mem::zeroed() };
let mut unit: i32 = 0;
p = end.wrapping_offset(-(1 as isize));
unsafe {
*p = 0;
}
unit = base;
loop {
if p == begin {
return 22i32.wrapping_neg();
}
let __v0: *mut u8 = p;
let __v1: *mut u8 = __v0.wrapping_offset(-(1 as isize));
p = __v1;
unsafe {
*__v1 = 97i32.wrapping_add(index % unit) as u8;
}
index = (index / unit).wrapping_sub(1);
if index < 0 {
break;
}
}
unsafe { memmove(begin as *mut (), p as *const (), (end.offset_from(p as *mut u8) as i64) as u64) };
unsafe { memcpy(buf as *mut (), prefix as *const (), strlen(prefix as *const u8)) };
0
}

/// sd_probe - called during driver initialization and whenever a
/// new scsi device is attached to the system. It is called once
/// for each scsi device (not just disks) present.
/// @sdp: pointer to device object
///
/// Returns 0 if successful (or not interested in this scsi device
/// (e.g. scanner)); 1 when there is an error.
///
/// Note: this function is invoked from the scsi mid-level.
/// This function sets up the mapping between a given
/// <host,channel,id,lun> (found in sdp) and new device name
/// (e.g. /dev/sda). More precisely it is the block device major
/// and minor number that is chosen here.
///
/// Assume sd_probe is not re-entrant (for time being)
/// Also think about sd_probe() and sd_remove() running coincidentally.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_probe")]
extern "C" fn sd_probe(sdp: *mut scsi_device) -> i32 {
let mut __slate_compound_17: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut error: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(error) as *mut i32;
let mut index: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut i32 = core::ptr::addr_of_mut!(index) as *mut i32;
let mut gd: core::mem::MaybeUninit<*mut gendisk> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut gendisk = core::ptr::addr_of_mut!(gd) as *mut *mut gendisk;
let mut sdkp: core::mem::MaybeUninit<*mut scsi_disk> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut *mut scsi_disk = core::ptr::addr_of_mut!(sdkp) as *mut *mut scsi_disk;
let mut dev: *mut device = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
dev = core::ptr::addr_of_mut!((*sdp).sdev_gendev);
let __slate_auto_bytes_2 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
scsi_autopm_get_device(sdp);
*__slate_slot_12 = 19i32.wrapping_neg();
if !(((*sdp).r#type as i32) != 0 && ((*sdp).r#type as i32) != 20 && ((*sdp).r#type as i32) != 7 && ((*sdp).r#type as i32) != 14) {
if IS_ENABLED!() == 0 && ((*sdp).r#type as i32) == 20 {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, sdp as *const scsi_device, core::ptr::null::<u8>(), c"Unsupported ZBC host-managed device.\n".as_ptr() as *const u8);
} else {
SCSI_LOG_HLQUEUE!();
*__slate_slot_12 = 12i32.wrapping_neg();
__obj_size = size_mul(core::mem::size_of::<scsi_disk>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_6 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_7 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_17 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_17)
});
alloc_tag_restore!();
} else {
*__slate_slot_7 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
}
*__slate_slot_15 = *__slate_slot_7 as *mut scsi_disk;
if !(*__slate_slot_15).is_null() {
*__slate_slot_14 = blk_mq_alloc_disk_for_queue((*sdp).request_queue, core::ptr::addr_of_mut!(sd_bio_compl_lkclass));
if !(*__slate_slot_14).is_null() {
*__slate_slot_13 = ida_alloc(core::ptr::addr_of_mut!(sd_index_ida), ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
if *__slate_slot_13 < 0 {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, sdp as *const scsi_device, core::ptr::null::<u8>(), c"sd_probe: memory exhausted.\n".as_ptr() as *const u8);
} else {
*__slate_slot_12 = sd_format_disk_name(c"sd".as_ptr() as *mut u8, *__slate_slot_13, (*(*__slate_slot_14)).disk_name.as_mut_ptr() as *mut u8, 32i32);
if *__slate_slot_12 != 0 {
sdev_prefix_printk(c"\x014".as_ptr() as *const u8, sdp as *const scsi_device, core::ptr::null::<u8>(), c"SCSI disk (sd) name length exceeded.\n".as_ptr() as *const u8);
ida_free(core::ptr::addr_of_mut!(sd_index_ida), *__slate_slot_13 as u32);
} else {
(*(*__slate_slot_15)).device = sdp;
(*(*__slate_slot_15)).disk = *__slate_slot_14;
(*(*__slate_slot_15)).index = *__slate_slot_13 as u32;
(*(*__slate_slot_15)).max_retries = 5;
atomic_set(core::ptr::addr_of_mut!((*(*__slate_slot_15)).openers), 0i32);
atomic_set(core::ptr::addr_of_mut!((*(*(*__slate_slot_15)).device).ioerr_cnt), 0i32);
if (*(*sdp).request_queue).rq_timeout == 0 {
if ((*sdp).r#type as i32) != 7 {
blk_queue_rq_timeout((*sdp).request_queue, 30000u32);
} else {
blk_queue_rq_timeout((*sdp).request_queue, 75000u32);
}
}
device_initialize(core::ptr::addr_of_mut!((*(*__slate_slot_15)).disk_dev));
(*(*__slate_slot_15)).disk_dev.parent = get_device(dev);
(*(*__slate_slot_15)).disk_dev.class = core::ptr::addr_of_mut!(sd_disk_class) as *const class;
dev_set_name(core::ptr::addr_of_mut!((*(*__slate_slot_15)).disk_dev), c"%s".as_ptr() as *const u8, dev_name(dev as *const device));
*__slate_slot_12 = device_add(core::ptr::addr_of_mut!((*(*__slate_slot_15)).disk_dev));
if *__slate_slot_12 != 0 {
put_device(core::ptr::addr_of_mut!((*(*__slate_slot_15)).disk_dev));
put_disk(*__slate_slot_14);
break '__join_0;
} else {
dev_set_drvdata(dev, *__slate_slot_15 as *mut ());
(*(*__slate_slot_14)).major = sd_major((*__slate_slot_13 & 240) >> 4i32);
(*(*__slate_slot_14)).first_minor = (*__slate_slot_13 & 15) << 4i32 | *__slate_slot_13 & 1048320;
(*(*__slate_slot_14)).minors = 16;
(*(*__slate_slot_14)).fops = core::ptr::addr_of!(sd_fops.0);
(*(*__slate_slot_14)).private_data = *__slate_slot_15 as *mut ();
// defaults, until the device tells us otherwise
(*sdp).sector_size = 512;
(*(*__slate_slot_15)).capacity = 0;
(*(*__slate_slot_15)).media_present = 1;
(*(*__slate_slot_15)).write_prot = 0;
(*(*__slate_slot_15)).__slate_bits_0.__set_cache_override(0u32);
(*(*__slate_slot_15)).__slate_bits_0.__set_WCE(0u32);
(*(*__slate_slot_15)).__slate_bits_0.__set_RCD(0u32);
(*(*__slate_slot_15)).__slate_bits_0.__set_ATO(0u32);
(*(*__slate_slot_15)).__slate_bits_0.__set_first_scan(1u32);
(*(*__slate_slot_15)).max_medium_access_timeouts = 2;
sd_revalidate_disk(*__slate_slot_14);
if ((*sdp).sector_size as u64) > _AC!() << 12i32 {
if sd_large_pool_create() != 0 {
*__slate_slot_12 = 12i32.wrapping_neg();
device_unregister(core::ptr::addr_of_mut!((*(*__slate_slot_15)).disk_dev));
put_disk(*__slate_slot_14);
break '__join_0;
}
}
if ((*sdp).__slate_bits_0.__get_removable() as i32) != 0 {
let __v5: *mut gendisk = *__slate_slot_14;
(*__v5).flags = (*__v5).flags | 1;
let __v2: *mut gendisk = *__slate_slot_14;
(*__v2).events = (((*__v2).events as i32) | 1) as u16;
(*(*__slate_slot_14)).event_flags = 3u16;
}
blk_pm_runtime_init((*sdp).request_queue, dev);
if ((*sdp).__slate_bits_0.__get_rpm_autosuspend() as i32) != 0 {
pm_runtime_set_autosuspend_delay(dev, (*(*sdp).host).rpm_autosuspend_delay);
}
*__slate_slot_12 = device_add_disk(dev, *__slate_slot_14, core::ptr::null_mut::<*const attribute_group>());
if *__slate_slot_12 != 0 {
device_unregister(core::ptr::addr_of_mut!((*(*__slate_slot_15)).disk_dev));
put_disk(*__slate_slot_14);
if ((*sdp).sector_size as u64) > _AC!() << 12i32 {
sd_large_pool_destroy();
break '__join_0;
} else {
break '__join_0;
}
} else {
if ((*(*__slate_slot_15)).__slate_bits_0.__get_security() as i32) != 0 {
(*(*__slate_slot_15)).opal_dev = core::ptr::null_mut::<opal_dev>();
if !unsafe { (*(*__slate_slot_15)).opal_dev }.is_null() {
if !unsafe { (*(*__slate_slot_15)).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*(*__slate_slot_15)).device as *const scsi_device, ((*(*(*__slate_slot_15)).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"supports TCG Opal\n".as_ptr() as *const u8);
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*(*__slate_slot_15)).device as *const scsi_device, core::ptr::null::<u8>(), c"supports TCG Opal\n".as_ptr() as *const u8);
}
}
}
if !unsafe { (*(*__slate_slot_15)).disk }.is_null() {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*(*__slate_slot_15)).device as *const scsi_device, ((*(*(*__slate_slot_15)).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Attached SCSI %sdisk\n".as_ptr() as *const u8, if ((*sdp).__slate_bits_0.__get_removable() as i32) != 0 { c"removable ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 });
} else {
sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*(*__slate_slot_15)).device as *const scsi_device, core::ptr::null::<u8>(), c"Attached SCSI %sdisk\n".as_ptr() as *const u8, if ((*sdp).__slate_bits_0.__get_removable() as i32) != 0 { c"removable ".as_ptr() as *mut u8 } else { c"".as_ptr() as *mut u8 });
}
scsi_autopm_put_device(sdp);
return 0;
}
}
}
}
put_disk(*__slate_slot_14);
}
kfree(*__slate_slot_15 as *const ());
}
}
}
}
scsi_autopm_put_device(sdp);
return *__slate_slot_12;
}
}

fn sd_start_stop_device(sdkp: *mut scsi_disk, start: i32) -> i32 {
let mut cmd: [u8; 6] = {
let mut __t0: [u8; 6] = unsafe { core::mem::zeroed() };
__t0[0] = 27u8;
__t0
};
// START_VALID
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut failure_defs: __SlateAlign16<[scsi_failure; 4]> = __SlateAlign16([{
let mut __t1: scsi_failure = unsafe { core::mem::zeroed() };
__t1.result = 2i32;
__t1.sense = 6u8;
__t1.asc = 41u8;
__t1.ascq = 0u8;
__t1
}, {
let mut __t2: scsi_failure = unsafe { core::mem::zeroed() };
__t2.result = 2i32;
__t2.sense = 6u8;
__t2.asc = 41u8;
__t2.ascq = 1u8;
__t2
}, {
let mut __t3: scsi_failure = unsafe { core::mem::zeroed() };
__t3.result = 2i32;
__t3.sense = 6u8;
__t3.asc = 41u8;
__t3.ascq = 2u8;
__t3
}, {
let mut __t4: scsi_failure = unsafe { core::mem::zeroed() };
__t4
}]);
// Power on, reset, or bus device reset occurred
// Power on occurred
// SCSI bus reset
let mut failures: scsi_failures = {
let mut __t5: scsi_failures = unsafe { core::mem::zeroed() };
__t5.total_allowed = 3i32;
__t5.failure_definitions = failure_defs.0.as_mut_ptr() as *mut scsi_failure;
__t5
};
let exec_args: scsi_exec_args = {
let mut __t6: scsi_exec_args = unsafe { core::mem::zeroed() };
__t6.sshdr = core::ptr::addr_of_mut!(sshdr);
__t6.req_flags = 4u32;
__t6.failures = core::ptr::addr_of_mut!(failures);
__t6
};
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
let mut res: i32 = 0;
if start != 0 {
let __v0: *mut u8 = (cmd.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 1) as u8;
}
}
// START
if ((unsafe { (*sdp).__slate_bits_0.__get_start_stop_pwr_cond() }) as i32) != 0 {
let __v3: *mut u8 = (cmd.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize);
unsafe {
*__v3 = ((*__v3 as i32) | if start != 0 { 16i32 } else { 48i32 }) as u8;
}
}
// Active or Standby
if scsi_device_online(sdp) == 0 {
return 19i32.wrapping_neg();
}
res = unsafe { scsi_execute_cmd(sdp, cmd.as_mut_ptr() as *const u8, 34u32, core::ptr::null_mut::<()>(), 0u32, 30000i32, (*sdkp).max_retries, core::ptr::addr_of!(exec_args)) };
if res != 0 {
sd_print_result(sdkp as *const scsi_disk, c"Start/Stop Unit failed".as_ptr() as *const u8, res);
let __v6: bool = res > 0 && scsi_sense_valid(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr);
if __v6 {
sd_print_sense_hdr(sdkp, core::ptr::addr_of_mut!(sshdr));
// 0x3a is medium not present
if (sshdr.asc as i32) == 58 {
res = 0;
}
}
}
// SCSI error codes must not go to the generic layer
if res != 0 {
return 5i32.wrapping_neg();
}
0
}

/// Send a SYNCHRONIZE CACHE instruction down to the device through
/// the normal SCSI command structure.  Wait for the command to
/// complete.
#[unsafe(link_section = ".text.slate_distinct.sd.sd_shutdown")]
extern "C" fn sd_shutdown(sdp: *mut scsi_device) {
let dev: *mut device = unsafe { core::ptr::addr_of_mut!((*sdp).sdev_gendev) };
let sdkp: *mut scsi_disk = dev_get_drvdata(dev as *const device) as *mut scsi_disk;
if sdkp.is_null() {
return;
}
// this can happen
if pm_runtime_suspended(dev) {
return;
}
if ((unsafe { (*sdkp).__slate_bits_0.__get_WCE() }) as i32) != 0 && (unsafe { (*sdkp).media_present }) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Synchronizing SCSI cache\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Synchronizing SCSI cache\n".as_ptr() as *const u8) };
}
sd_sync_cache(sdkp);
}
if ((unsafe { system_state }) as u32) != 6 && ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_manage_system_start_stop() }) as i32) != 0 || ((unsafe { system_state }) as u32) == 5 && ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_manage_shutdown() }) as i32) != 0 || ((unsafe { system_state }) as u32) == 3 && ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_manage_runtime_start_stop() }) as i32) != 0 || ((unsafe { system_state }) as u32) == 6 && ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_manage_restart() }) as i32) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Stopping disk\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Stopping disk\n".as_ptr() as *const u8) };
}
sd_start_stop_device(sdkp, 0i32);
}
}

/// sd_remove - called whenever a scsi disk (previously recognized by
/// sd_probe) is detached from the system. It is called (potentially
/// multiple times) during sd module unload.
/// @sdp: pointer to device object
///
/// Note: this function is invoked from the scsi mid-level.
/// This function potentially frees up a device name (e.g. /dev/sdc)
/// that could be re-used by a subsequent sd_probe().
/// This function is not called when the built-in sd driver is "exit-ed".
#[unsafe(link_section = ".text.slate_distinct.sd.sd_remove")]
extern "C" fn sd_remove(sdp: *mut scsi_device) {
let dev: *mut device = unsafe { core::ptr::addr_of_mut!((*sdp).sdev_gendev) };
let sdkp: *mut scsi_disk = dev_get_drvdata(dev as *const device) as *mut scsi_disk;
unsafe { scsi_autopm_get_device((*sdkp).device) };
unsafe { device_del(core::ptr::addr_of_mut!((*sdkp).disk_dev)) };
unsafe { del_gendisk((*sdkp).disk) };
if !unsafe { (*sdkp).suspended } {
sd_shutdown(sdp);
}
unsafe { put_disk((*sdkp).disk) };
if ((unsafe { (*sdp).sector_size }) as u64) > _AC!() << 12i32 {
sd_large_pool_destroy();
}
}

fn sd_do_start_stop(sdev: *mut scsi_device, runtime: bool) -> bool {
((unsafe { (*sdev).__slate_bits_0.__get_manage_system_start_stop() }) as i32) != 0 && !runtime || ((unsafe { (*sdev).__slate_bits_0.__get_manage_runtime_start_stop() }) as i32) != 0 && runtime
}

fn sd_suspend_common(dev: *mut device, runtime: bool) -> i32 {
let sdkp: *mut scsi_disk = dev_get_drvdata(dev as *const device) as *mut scsi_disk;
let mut ret: i32 = 0;
if sdkp.is_null() {
return 0;
}
// E.g.: runtime suspend following sd_remove()
if ((unsafe { (*sdkp).__slate_bits_0.__get_WCE() }) as i32) != 0 && (unsafe { (*sdkp).media_present }) != 0 {
if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_silence_suspend() }) as i32) == 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Synchronizing SCSI cache\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Synchronizing SCSI cache\n".as_ptr() as *const u8) };
}
}
ret = sd_sync_cache(sdkp);
// ignore OFFLINE device
if ret == 19i32.wrapping_neg() {
return 0;
}
if ret != 0 {
return ret;
}
}
if sd_do_start_stop(unsafe { (*sdkp).device }, runtime) {
if ((unsafe { (*(*sdkp).device).__slate_bits_0.__get_silence_suspend() }) as i32) == 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Stopping disk\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Stopping disk\n".as_ptr() as *const u8) };
}
}
// an error is not worth aborting a system sleep
ret = sd_start_stop_device(sdkp, 0i32);
if !runtime {
ret = 0;
}
}
if ret == 0 {
unsafe {
(*sdkp).suspended = true;
}
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_suspend_system")]
extern "C" fn sd_suspend_system(dev: *mut device) -> i32 {
if pm_runtime_suspended(dev) {
return 0;
}
sd_suspend_common(dev, false)
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_suspend_runtime")]
extern "C" fn sd_suspend_runtime(dev: *mut device) -> i32 {
sd_suspend_common(dev, true)
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_resume")]
extern "C" fn sd_resume(dev: *mut device) -> i32 {
let sdkp: *mut scsi_disk = dev_get_drvdata(dev as *const device) as *mut scsi_disk;
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Starting disk\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Starting disk\n".as_ptr() as *const u8) };
}
if opal_unlock_from_suspend(unsafe { (*sdkp).opal_dev }) {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"OPAL unlock failed\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"OPAL unlock failed\n".as_ptr() as *const u8) };
}
return 5i32.wrapping_neg();
}
0
}

fn sd_resume_common(dev: *mut device, runtime: bool) -> i32 {
let sdkp: *mut scsi_disk = dev_get_drvdata(dev as *const device) as *mut scsi_disk;
let mut ret: i32 = 0;
if sdkp.is_null() {
return 0;
}
// E.g.: runtime resume at the start of sd_probe()
if !sd_do_start_stop(unsafe { (*sdkp).device }, runtime) {
unsafe {
(*sdkp).suspended = false;
}
return 0;
}
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Starting disk\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Starting disk\n".as_ptr() as *const u8) };
}
ret = sd_start_stop_device(sdkp, 1i32);
if ret == 0 {
sd_resume(dev);
unsafe {
(*sdkp).suspended = false;
}
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_resume_system")]
extern "C" fn sd_resume_system(dev: *mut device) -> i32 {
if pm_runtime_suspended(dev) {
let sdkp: *mut scsi_disk = dev_get_drvdata(dev as *const device) as *mut scsi_disk;
let sdp: *mut scsi_device = if !sdkp.is_null() { unsafe { (*sdkp).device } } else { core::ptr::null_mut::<scsi_device>() };
if !sdp.is_null() && ((unsafe { (*sdp).__slate_bits_0.__get_force_runtime_start_on_system_start() }) as i32) != 0 {
pm_request_resume(dev);
}
return 0;
}
sd_resume_common(dev, false)
}

#[unsafe(link_section = ".text.slate_distinct.sd.sd_resume_runtime")]
extern "C" fn sd_resume_runtime(dev: *mut device) -> i32 {
let sdkp: *mut scsi_disk = dev_get_drvdata(dev as *const device) as *mut scsi_disk;
if sdkp.is_null() {
return 0;
}
// E.g.: runtime resume at the start of sd_probe()
let sdp: *mut scsi_device = unsafe { (*sdkp).device };
if ((unsafe { (*sdp).__slate_bits_0.__get_ignore_media_change() }) as i32) != 0 {
// clear the device's sense data
let exec_args: scsi_exec_args = {
let mut __t0: scsi_exec_args = unsafe { core::mem::zeroed() };
__t0.req_flags = 4u32;
__t0
};
if (unsafe { scsi_execute_cmd(sdp, core::ptr::addr_of!(cmd__4404) as *const u8, 34u32, core::ptr::null_mut::<()>(), 0u32, (*(*sdp).request_queue).rq_timeout as i32, 1i32, core::ptr::addr_of!(exec_args)) }) != 0 {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"Failed to clear sense data\n".as_ptr() as *const u8) };
} else {
unsafe { sdev_prefix_printk(c"\x015".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"Failed to clear sense data\n".as_ptr() as *const u8) };
}
}
}
sd_resume_common(dev, true)
}

static cmd__4404: [u8; 10] = {
let mut __t0: [u8; 10] = unsafe { core::mem::zeroed() };
__t0[0] = 3u8;
__t0
};

static sd_pm_ops: __SlateConst<dev_pm_ops> = __SlateConst({
let mut __t0: dev_pm_ops = unsafe { core::mem::zeroed() };
__t0.suspend = Some(sd_suspend_system);
__t0.resume = Some(sd_resume_system);
__t0.poweroff = Some(sd_suspend_system);
__t0.restore = Some(sd_resume_system);
__t0.runtime_suspend = Some(sd_suspend_runtime);
__t0.runtime_resume = Some(sd_resume_runtime);
__t0
});

static mut sd_template: scsi_driver = scsi_driver { gendrv: {
let mut __t0: device_driver = unsafe { core::mem::zeroed() };
__t0.name = (b"sd\0".as_ptr() as *mut u8) as *const u8;
__t0.probe_type = 1u32;
__t0.pm = unsafe { core::ptr::addr_of!(sd_pm_ops.0) };
__t0
}, probe: Some(sd_probe), remove: Some(sd_remove), shutdown: Some(sd_shutdown), resume: Some(sd_resume), rescan: Some(sd_rescan), init_command: Some(sd_init_command), uninit_command: Some(sd_uninit_command), done: Some(sd_done), eh_action: Some(sd_eh_action), eh_reset: Some(sd_eh_reset) };

/// init_sd - entry point for this driver (both when built in or when
/// a module).
///
/// Note: this function registers this driver with the scsi mid-level.
#[unsafe(link_section = ".init.text")]
extern "C" fn init_sd() -> i32 {
let mut err: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(err) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut majors: i32 = unsafe { core::mem::zeroed() };
unsafe {
majors = 0;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
SCSI_LOG_HLQUEUE!();
*__slate_slot_8 = 0;
while *__slate_slot_8 < 16 {
if __register_blkdev(sd_major(*__slate_slot_8) as u32, c"sd".as_ptr() as *const u8, Some(sd_default_probe)) == 0 {
majors = majors.wrapping_add(1);
}
let __v4: i32 = *__slate_slot_8;
*__slate_slot_8 = __v4.wrapping_add(1);
}
if majors == 0 {
return 19i32.wrapping_neg();
} else {
*__slate_slot_7 = class_register(core::ptr::addr_of_mut!(sd_disk_class) as *const class);
if *__slate_slot_7 == 0 {
sd_page_pool = mempool_create_page_pool__2!(2i32);
if unsafe { sd_page_pool }.is_null() {
while false {
}
_printk(c"\x013sd: can't init discard page pool\n".as_ptr() as *const u8);
*__slate_slot_7 = 12i32.wrapping_neg();
} else {
*__slate_slot_7 = scsi_register_driver!(core::ptr::addr_of_mut!(sd_template));
if *__slate_slot_7 != 0 {
mempool_destroy(sd_page_pool);
} else {
return 0;
}
}
class_unregister(core::ptr::addr_of_mut!(sd_disk_class) as *const class);
}
*__slate_slot_8 = 0;
while *__slate_slot_8 < 16 {
unregister_blkdev(sd_major(*__slate_slot_8) as u32, c"sd".as_ptr() as *const u8);
let __v1: i32 = *__slate_slot_8;
*__slate_slot_8 = __v1.wrapping_add(1);
}
return *__slate_slot_7;
}
}
}

/// exit_sd - exit point for this driver (when it is a module).
///
/// Note: this function unregisters this driver from the scsi mid-level.
#[unsafe(link_section = ".exit.text")]
extern "C" fn exit_sd() {
SCSI_LOG_HLQUEUE!();
scsi_unregister_driver!(unsafe { core::ptr::addr_of_mut!(sd_template) });
unsafe { mempool_destroy(sd_page_pool) };
if !unsafe { sd_large_page_pool }.is_null() {
unsafe { mempool_destroy(sd_large_page_pool) };
}
unsafe { class_unregister(core::ptr::addr_of_mut!(sd_disk_class) as *const class) };
for i in 0i32..16i32 {
unsafe { unregister_blkdev(sd_major(i) as u32, c"sd".as_ptr() as *const u8) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_init_sd_675: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(init_sd)) };

#[unsafe(link_section = ".exitcall.exit")]
static mut __exitcall_exit_sd: Option<unsafe extern "C" fn()> = Some(exit_sd);

#[unsafe(no_mangle)]
extern "C" fn sd_print_sense_hdr(sdkp: *mut scsi_disk, sshdr: *mut scsi_sense_hdr) {
unsafe { scsi_print_sense_hdr((*sdkp).device as *const scsi_device, (if !unsafe { (*sdkp).disk }.is_null() { (*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8 } else { core::ptr::null_mut::<u8>() }) as *const u8, sshdr as *const scsi_sense_hdr) };
}

#[unsafe(no_mangle)]
extern "C" fn sd_print_result(sdkp: *const scsi_disk, msg: *const u8, result: i32) {
let hb_string: *const u8 = unsafe { scsi_hostbyte_string(result) };
if !hb_string.is_null() {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"%s: Result: hostbyte=%s driverbyte=%s\n".as_ptr() as *const u8, msg, if !hb_string.is_null() { hb_string } else { c"invalid".as_ptr() as *const u8 }, c"DRIVER_OK".as_ptr() as *mut u8) };
} else {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"%s: Result: hostbyte=%s driverbyte=%s\n".as_ptr() as *const u8, msg, if !hb_string.is_null() { hb_string } else { c"invalid".as_ptr() as *const u8 }, c"DRIVER_OK".as_ptr() as *mut u8) };
}
} else {
if !unsafe { (*sdkp).disk }.is_null() {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, ((*(*sdkp).disk).disk_name.as_mut_ptr() as *mut u8) as *const u8, c"%s: Result: hostbyte=0x%02x driverbyte=%s\n".as_ptr() as *const u8, msg, host_byte!(result), c"DRIVER_OK".as_ptr() as *mut u8) };
} else {
unsafe { sdev_prefix_printk(c"\x016".as_ptr() as *const u8, (*sdkp).device as *const scsi_device, core::ptr::null::<u8>(), c"%s: Result: hostbyte=0x%02x driverbyte=%s\n".as_ptr() as *const u8, msg, host_byte!(result), c"DRIVER_OK".as_ptr() as *mut u8) };
}
}
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_bvec_virt_1: [u8; 1] = *b"\0";

static __slate_asm_string_bvec_virt_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/bvec.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_sd_init_command_3: [u8; 1] = *b"\0";

static __slate_asm_string_sd_init_command_4: [u8; 47] = *b"/home/takashi/c-corpus/linux/drivers/scsi/sd.c\0";

static __slate_asm_string_sd_major_0: [u8; 1] = *b"\0";

static __slate_asm_string_sd_major_1: [u8; 47] = *b"/home/takashi/c-corpus/linux/drivers/scsi/sd.c\0";

static __slate_asm_string_sd_setup_rw6_cmnd_1: [u8; 1] = *b"\0";

static __slate_asm_string_sd_setup_rw6_cmnd_2: [u8; 47] = *b"/home/takashi/c-corpus/linux/drivers/scsi/sd.c\0";


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
//! libata-scsi.c - helper library for ATA
//!
//! Copyright 2003-2004 Red Hat, Inc.  All rights reserved.
//! Copyright 2003-2004 Jeff Garzik
//!
//! libata documentation is available via 'make {ps|pdf}docs',
//! as Documentation/driver-api/libata.rst
//!
//! Hardware documentation available from
//! - http://www.t10.org/
//! - http://www.t13.org/
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
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! restart by calling sys_restart_syscall
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
//! An exchange-type operation, which takes a value and a pointer, and
//! returns the old value.
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
//! mutex_init - initialize the mutex
//! @mutex: the mutex to be initialized
//!
//! Initialize the mutex to unlocked state.
//!
//! It is not allowed to initialize an already locked mutex.
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
//! Suits me
//! __MAX_NR_ZONES
//! percpu_counter batch for local add or sub
//! Desired design of maximum size and alignment (see RFC2553)
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
//! number of env pointers
//! kunmap_local - Unmap a page mapped via kmap_local_page().
//! @__addr: An address within the page mapped
//!
//! @__addr can be any address within the mapped page.  Commonly it is the
//! address return from kmap_local_page(), but it can also include offsets.
//!
//! Unmapping should be done in the reverse order of the mapping.  See
//! kmap_local_page() for details.
//! Dummy dev_printk for disabled debugging statements to use whilst maintaining
//! gcc's format checking.
//! Generate SD flag indexes
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
//! iterate over multi-page bvec
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! values for service action in
//! Special value for scanning to specify scanning or rescanning of all
//! possible channels, (target) ids, or luns on a given shost.
//! See the comment in asm-generic/io.h about ioremap_np().
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! MAX_COMMAND_SIZE is:
//! The longest fixed-length SCSI CDB as per the SCSI standard.
//! fixed-length means: commands that their size can be determined
//! by their opcode and the CDB does not carry a length specifier, (unlike
//! the VARIABLE_LENGTH_CMD(0x7f) command). This is actually not exactly
//! true and the SCSI standard also defines extended commands and
//! vendor specific commands that can be bigger than 16 bytes. The kernel
//! will support these using the same infrastructure used for VARLEN CDB's.
//! So in effect MAX_COMMAND_SIZE means the maximum size command scsi-ml
//! supports without specifying a cmd_len by ULD's
//! identify no tag in use
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
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
//! should be >= TASK_COMM_LEN
//! in linux/sched.h
//! hlist_nulls_first_rcu - returns the first element of the hash list.
//! @head: the head of the list.
//! define the enumeration of all cgroup subsystems
//! To avoid confusing the compiler (and generating warnings) with code
//! that attempts to access what would be a 0-element array (i.e. sized
//! to a potentially empty array when CGROUP_SUBSYS_COUNT == 0), this
//! constant expression can be added.
//! 31 pointers + header align the folio_batch structure to a power of two
//! must be exactly four chars
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
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
// round_down - round down to next specified power of 2
// @x: the value to round
// @y: multiple to round down to (must be a power of 2)
//
// Rounds @x down to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding down, use rounddown() below.
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
// "fsrm" Fast Short Rep Mov
// "serialize" SERIALIZE instruction
// LFENCE always serializing / synchronizes RDTSC
// BUG word(s)
// "apic_c1e" AMD Erratum 400
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
// Not a typewriter
// Text file busy
// No space left on device
// Math result not representable
// Invalid system call number
// Operation would block
// No message of desired type
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
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
// min_not_zero - return the minimum that is _not_ zero, unless both are zero
// @x: value1
// @y: value2
// clamp - return a value clamped to a given range with typechecking
// @val: current value
// @lo: lowest allowable value
// @hi: highest allowable value
//
// This macro checks @val/@lo/@hi to make sure they have compatible
// signedness.
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
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// Same as rb_first(), but O(1)
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
// Don't call further
// Implementation specific max size
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
// values for ZBC_IN
// values for ZBC_OUT
// Values for T10/04-262r7
// 16-byte pass-thru
// 12-byte pass-thru
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
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// DWORD 5
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
/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__24 {
() => {
((core::mem::size_of::<[ata_scsi_cmd; 40]>() as u64) / (core::mem::size_of::<ata_scsi_cmd>() as u64)).wrapping_add(__must_be_array__89!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__87 {
() => {
((core::mem::size_of::<[ata_scsi_cmd; 40]>() as u64) / (core::mem::size_of::<ata_scsi_cmd>() as u64)).wrapping_add(__must_be_array__37!() as u64)
};
}

/// ARRAY_SIZE - get the number of elements in array @arr
/// @arr: array to be sized
macro_rules! ARRAY_SIZE__116 {
() => {
((core::mem::size_of::<[ata_scsi_cmd; 40]>() as u64) / (core::mem::size_of::<ata_scsi_cmd>() as u64)).wrapping_add(__must_be_array__106!() as u64)
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

macro_rules! COMMAND_SIZE {
($opcode:expr) => {
unsafe { scsi_command_size_tbl[((($opcode) as i32) >> 5i32 & 7) as usize] }
};
}

macro_rules! IOPRIO_PRIO_CLASS {
($ioprio:expr) => {
(($ioprio) as i32) >> 13i32 & 7i32
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

macro_rules! RELOC_HIDE {
($ptr:expr, $off:expr) => {
(($ptr) as u64).wrapping_add(($off) as u64) as *mut ()
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
macro_rules! __acquire__2 {
($x:expr) => {
__acquire_ctx_lock__2(($x) as *const spinlock);
};
}

macro_rules! __chk_user_ptr {
() => {
};
}

macro_rules! __cmp_once__2 {
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
macro_rules! __get_unaligned_t {
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

macro_rules! __must_be_array__37 {
() => {
core::mem::size_of::<__SlateRecord_libata_scsi_c_3863>() as i32
};
}

macro_rules! __must_be_array__89 {
() => {
core::mem::size_of::<__SlateRecord_libata_scsi_c_3959>() as i32
};
}

macro_rules! __must_be_array__106 {
() => {
core::mem::size_of::<__SlateRecord_libata_scsi_c_3877>() as i32
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

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release__2 {
($x:expr) => {
__release_ctx_lock__1(($x) as *const raw_spinlock);
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
if is_signed_type__2!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
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

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use!($ux) & __sign_use!($uy)
};
}

macro_rules! __types_ok {
($ux:expr, $uy:expr) => {
__sign_use__2!($ux) & __sign_use__2!($uy)
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

macro_rules! ata_id_is_locked {
($id:expr) => {
((unsafe { *($id.as_mut_ptr() as *mut u16).wrapping_offset(128 as isize) }) as i32) & 7 == 7
};
}

macro_rules! ata_id_queue_depth {
($id:expr) => {
(((unsafe { *($id.as_mut_ptr() as *mut u16).wrapping_offset(75 as isize) }) as i32) & 31).wrapping_add(1)
};
}

macro_rules! ata_id_removable {
($id:expr) => {
((unsafe { *($id.as_mut_ptr() as *mut u16).wrapping_offset(0 as isize) }) as i32) & 128i32
};
}

/// container_of - cast a member of a structure out to the containing structure
/// @ptr:	the pointer to the member.
/// @type:	the type of the container struct this is embedded in.
/// @member:	the name of the member within the struct.
///
/// WARNING: any const qualifier of @ptr is lost.
/// Do not use container_of() in new code.
macro_rules! container_of__97 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__106!($type) as isize)) }) as *mut ()) as *mut $type;
__v
}
};
}

macro_rules! current_thread_info {
() => {
get_current() as *mut thread_info
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
macro_rules! do_div__4 {
($n:expr, $base:expr) => {
let mut __v: u32;
let __base: u32 = ($base) as u32;
let __rem: u32 = ((($n) as u64) % (__base as u64)) as u32;
$n = (($n) as u64) / (__base as u64);
__v = __rem;
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
macro_rules! instrument_put_user__3 {
($from:expr, $ptr:expr) => {
kmsan_copy_to_user(($ptr) as *mut (), core::ptr::addr_of_mut!($from) as *const (), core::mem::size_of::<u64>() as u64, 0u64);
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__3 {
() => {
(1i32.wrapping_neg() as u32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
};
}

macro_rules! kcsan_check_atomic_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 5i32);
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

macro_rules! likely__5 {
($x:expr) => {
!$x.is_null() as i64
};
}

macro_rules! lockdep_assert_held__2 {
($l:expr) => {
__assume_ctx_lock__2(($l) as *const spinlock);
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__2!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once!($type, $x, $y)
};
}

macro_rules! offsetof__2 {
() => {
40u64
};
}

macro_rules! offsetof__106 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, deferred_qc_work) as u64
};
}

macro_rules! offsetof__120 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, scsi_rescan_task.work) as u64
};
}

macro_rules! offsetof__52 {
() => {
456u64
};
}

macro_rules! offsetof__150 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, hotplug_task.work) as u64
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

macro_rules! sector_div {
($a:expr, $b:expr) => {
do_div__4!($a, $b);
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

/// time_after - returns true if the time a is after time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Do this with "<0" and ">=0" to only test the sign of the result. A
/// good compiler would generate better code (and a really good compiler
/// wouldn't care). Gcc is currently neither.
///
/// Return: %true is time a is after time b, otherwise %false.
macro_rules! time_after__2 {
($a:expr, $b:expr) => {
{
let __v: bool = typecheck__2!(u64) != 0 && typecheck__2!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

macro_rules! to_scsi_device {
($d:expr) => {
{
let mut __v: *mut scsi_device;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof__52!() as isize)) }) as *mut ()) as *mut scsi_device;
__v
}
};
}

macro_rules! to_scsi_target {
($d:expr) => {
{
let mut __v: *mut scsi_target;
__v = ((unsafe { ((($d) as *mut ()) as *const u8).wrapping_offset(-(offsetof__2!() as isize)) }) as *mut ()) as *mut scsi_target;
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
    static mut cpuinfo_op: seq_operations;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut system_highpri_wq: *mut workqueue_struct;
    static mut system_dfl_long_wq: *mut workqueue_struct;
    static mut sighand_cachep: *mut kmem_cache;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut scsi_command_size_tbl: [u8; 8];
    static mut kstat: kernel_stat;
    static mut atapi_passthru16: i32;
    static mut libata_allow_tpm: i32;
    static mut ata_scsi_transportt: scsi_transport_template;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn strncmp(arg0: *const u8, arg1: *const u8, arg2: u64) -> i32;
    fn kstrtoint(arg0: *const u8, arg1: u32, arg2: *mut i32) -> i32;
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn complete(arg0: *mut completion);
    fn __msecs_to_jiffies(arg0: u32) -> u64;
    fn round_jiffies_relative(arg0: u64) -> u64;
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn queue_delayed_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut delayed_work, arg3: u64) -> bool;
    fn cancel_work(arg0: *mut work_struct) -> bool;
    fn __bad_copy_from();
    fn __bad_copy_to();
    fn __copy_overflow(arg0: i32, arg1: u64);
    fn _copy_from_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn _copy_to_user(arg0: *mut (), arg1: *const (), arg2: u64) -> u64;
    fn capable(arg0: i32) -> bool;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_361();
    fn __compiletime_assert_386();
    fn kfree(arg0: *const ());
    fn __compiletime_assert_392();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn sysfs_emit(arg0: *mut u8, arg1: *const u8, ...) -> i32;
    fn device_link_add(arg0: *mut device, arg1: *mut device, arg2: u32) -> *mut device_link;
    fn device_link_remove(arg0: *mut (), arg1: *mut device);
    fn scsi_set_sense_information(arg0: *mut u8, arg1: i32, arg2: u64) -> i32;
    fn scsi_set_sense_field_pointer(arg0: *mut u8, arg1: i32, arg2: u16, arg3: u8, arg4: bool) -> i32;
    fn scsi_sense_desc_find(arg0: *const u8, arg1: i32, arg2: i32) -> *const u8;
    fn sg_copy_from_buffer(arg0: *mut scatterlist, arg1: u32, arg2: *const (), arg3: u64) -> u64;
    fn sg_copy_to_buffer(arg0: *mut scatterlist, arg1: u32, arg2: *mut (), arg3: u64) -> u64;
    fn sg_miter_start(arg0: *mut sg_mapping_iter, arg1: *mut scatterlist, arg2: u32, arg3: u32);
    fn sg_miter_next(arg0: *mut sg_mapping_iter) -> bool;
    fn sg_miter_stop(arg0: *mut sg_mapping_iter);
    fn scsi_host_alloc(arg0: *const scsi_host_template, arg1: i32) -> *mut Scsi_Host;
    fn scsi_add_host_with_dma(arg0: *mut Scsi_Host, arg1: *mut device, arg2: *mut device) -> i32;
    fn scsi_resume_device(arg0: *mut scsi_device) -> i32;
    fn scsi_rescan_device(arg0: *mut scsi_device) -> i32;
    fn scsi_remove_host(arg0: *mut Scsi_Host);
    fn __scsi_add_device(arg0: *mut Scsi_Host, arg1: u32, arg2: u32, arg3: u64, arg4: *mut ()) -> *mut scsi_device;
    fn scsi_remove_device(arg0: *mut scsi_device);
    fn scsi_device_get(arg0: *mut scsi_device) -> i32;
    fn scsi_device_put(arg0: *mut scsi_device);
    fn scsi_change_queue_depth(arg0: *mut scsi_device, arg1: i32) -> i32;
    fn scsi_device_set_state(arg0: *mut scsi_device, arg1: u32) -> i32;
    fn sdev_evt_send_simple(arg0: *mut scsi_device, arg1: u32, arg2: u32);
    fn scsi_execute_cmd(arg0: *mut scsi_device, arg1: *const u8, arg2: u32, arg3: *mut (), arg4: u32, arg5: i32, arg6: i32, arg7: *const scsi_exec_args) -> i32;
    fn scsi_done(arg0: *mut scsi_cmnd);
    fn scsi_build_sense(arg0: *mut scsi_cmnd, arg1: i32, arg2: u8, arg3: u8, arg4: u8);
    fn msleep(arg0: u32);
    fn cdrom_multisession(arg0: *mut cdrom_device_info, arg1: *mut cdrom_multisession) -> i32;
    fn atapi_cmd_type(arg0: u8) -> i32;
    fn ata_sg_init(arg0: *mut ata_queued_cmd, arg1: *mut scatterlist, arg2: u32);
    fn ata_id_string(arg0: *const u16, arg1: *mut u8, arg2: u32, arg3: u32);
    fn ata_acpi_gtm(arg0: *mut ata_port, arg1: *mut ata_acpi_gtm) -> i32;
    fn ata_port_schedule_eh(arg0: *mut ata_port);
    fn ata_port_wait_eh(arg0: *mut ata_port);
    fn ata_link_next(arg0: *mut ata_link, arg1: *mut ata_port, arg2: u32) -> *mut ata_link;
    fn ata_dev_next(arg0: *mut ata_device, arg1: *mut ata_link, arg2: u32) -> *mut ata_device;
    fn __compiletime_assert_601();
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn ata_build_rw_tf(arg0: *mut ata_queued_cmd, arg1: u64, arg2: u32, arg3: u32, arg4: i32, arg5: i32) -> i32;
    fn ata_tf_read_block(arg0: *const ata_taskfile, arg1: *mut ata_device) -> u64;
    fn ata_dev_power_init_tf(arg0: *mut ata_device, arg1: *mut ata_taskfile, arg2: bool) -> bool;
    fn ata_qc_free(arg0: *mut ata_queued_cmd);
    fn ata_qc_issue(arg0: *mut ata_port, arg1: *mut ata_queued_cmd);
    fn atapi_check_dma(arg0: *mut ata_queued_cmd) -> i32;
    fn ata_adapter_is_online(arg0: *mut ata_port) -> bool;
    fn ata_acpi_dev_manage_restart(arg0: *mut ata_device) -> bool;
    fn libata_transport_exit();
    fn __compiletime_assert_685();
    fn __compiletime_assert_688();
    fn __compiletime_assert_691();
    fn __compiletime_assert_705();
    fn __compiletime_assert_710();
    fn __compiletime_assert_718();
    fn __compiletime_assert_719();
    fn __compiletime_assert_724();
    fn __compiletime_assert_727();
    fn __compiletime_assert_730();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.bswap.i16"]
    fn __slate_intrinsic_1a0c079c71348b00(arg0: u16) -> u16;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.objectsize.i64.p0"]
    fn __slate_intrinsic_2913c0303de876b3(arg0: *const (), arg1: bool, arg2: bool, arg3: bool) -> u64;
    #[link_name = "llvm.bswap.i64"]
    fn __slate_intrinsic_5c27152fac4a123f(arg0: u64) -> u64;
    #[link_name = "llvm.is.constant.i64"]
    fn __slate_intrinsic_e1c548fa55baa574(arg0: i64) -> bool;
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
struct static_call_site {
    addr: i32,
    key: i32,
}


// __ADDRESSABLE() is used to ensure the key symbol doesn't get stripped from
// the symbol table so that objtool can reference it when it generates the
// .static_call_sites section.
#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
struct lockdep_map {
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
fn __acquire_ctx_lock__1(var: &raw_spinlock) {
}

#[inline(always)]
fn __release_ctx_lock__1(var: *const raw_spinlock) {
}

#[inline(always)]
fn __assume_ctx_lock__1(var: &raw_spinlock) {
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

fn kmsan_copy_to_user(to: *mut (), from: *const (), to_copy: u64, left: u64) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
#[inline(always)]
fn set_bit(nr: i64, addr: *mut u64) {
instrument_atomic_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_set_bit(nr, addr);
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits124U0,
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
    __slate_bits_0: __slate_bits::__SlateBits129U0,
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
struct perf_event {
}


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
    __slate_bits_0: __slate_bits::__SlateBits130U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_head {
    first: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
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
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[inline(always)]
fn __acquire_ctx_lock__2(var: *const spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn __release_ctx_lock__2(var: &spinlock) {
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
fn __assume_ctx_lock__2(var: *const spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct spinlock {
    __slate_anon_0: __SlateRecord_spinlock_types_h_18,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct futex_private_hash {
}


#[inline(always)]
fn __acquire_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __release_ctx_lock__3(var: &rwlock) {
}

#[inline(always)]
fn __assume_ctx_lock__3(var: &rwlock) {
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
struct rwlock {
    raw_lock: qrwlock,
}


#[inline(always)]
fn __acquire_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __release_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __assume_ctx_lock__4(var: &mutex) {
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
struct hrtimer_clock_base {
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
struct seccomp_filter {
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
struct timerqueue_linked_node {
    node: rb_node_linked,
    expires: i64,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_context {
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
struct sched_mm_cid {
    active: u32,
    cid: u32,
    node: hlist_node,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
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
struct rq {
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


#[inline(always)]
fn __acquire_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __release_ctx_lock__5(var: &seqlock) {
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
struct unwind_task_info {
    unwind_mask: __SlateRecord_types_h_197,
    cache: *mut unwind_cache,
    work: callback_head,
    id: unwind_task_id,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
union __SlateRecord_sched_h_826 {
    ipi_mask_ptr: *mut cpumask,
    ipi_mask_val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sched_class {
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
    __slate_bits_0: __slate_bits::__SlateBits286U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
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
struct vm_struct {
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
    __slate_bits_0: __slate_bits::__SlateBits77U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits77U1,
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
    __slate_bits_2: __slate_bits::__SlateBits77U2,
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
// This thread should not be frozen
// I am kswapd
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
struct rhash_head {
    next: *mut rhash_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kref {
    refcount: refcount_struct,
}


#[inline(always)]
fn __acquire_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[inline(always)]
fn __release_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn __assume_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[inline(always)]
fn __release_ctx_lock__7(var: &rw_semaphore) {
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
struct uprobe_xol_ops {
}


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
struct delayed_work {
    work: work_struct,
    timer: timer_list,
    wq: *mut workqueue_struct,
    cpu: i32,
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
struct rcu_work {
    work: work_struct,
    rcu: callback_head,
    wq: *mut workqueue_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
}


fn queue_work(wq: *mut workqueue_struct, work: *mut work_struct) -> bool {
unsafe { queue_work_on(64i32, wq, work) }
}

fn queue_delayed_work(wq: *mut workqueue_struct, dwork: *mut delayed_work, delay: u64) -> bool {
unsafe { queue_delayed_work_on(64i32, wq, dwork, delay) }
}

fn schedule_delayed_work(dwork: *mut delayed_work, delay: u64) -> bool {
queue_delayed_work(unsafe { system_percpu_wq }, dwork, delay)
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


#[inline(always)]
fn __acquire_ctx_lock__8(var: &srcu_struct) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
}


#[inline(always)]
fn __release_ctx_lock__8(var: &srcu_struct) {
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
fn __assume_ctx_lock__8(var: &srcu_struct) {
}

// Values for SRCU Tree srcu_data ->srcu_reader_flavor, but also used by rcutorture.
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
#[repr(C)]
#[derive(Clone, Copy)]
struct srcu_ctr {
    srcu_locks: __SlateRecord_types_h_197,
    srcu_unlocks: __SlateRecord_types_h_197,
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
struct return_consumer {
    cookie: u64,
    id: u64,
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
struct folio_queue {
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
union __SlateRecord_uio_h_78 {
    nr_segs: u64,
    folioq_slot: u8,
    xarray_start: i64,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
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
core::arch::asm!("180: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 180b - ., 3; .popsection\n# {0}", const 180 as i32, options(nostack, att_syntax));
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
core::arch::asm!("181: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 181b - ., 4; .popsection\n# {0}", const 181 as i32, options(nostack, att_syntax));
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
struct kmem_cache {
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
struct mnt_idmap {
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


#[inline(always)]
fn __acquire_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __release_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __assume_ctx_lock__9(var: &local_lock) {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct local_lock {
}


#[inline(always)]
fn __acquire_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


#[inline(always)]
fn __release_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
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
struct xarray {
    xa_lock: spinlock,
    xa_flags: u32,
    xa_head: *mut (),
}


// We encode errnos in the xas->xa_node.  If an error has happened, we need to
// drop the lock to fix it, and once we've done so the xa_state is invalid.
// XA_STATE() - Declare an XArray operation state.
// @name: Name of this operation state (usually xas).
// @array: Array to operate on.
// @index: Initial index of interest.
//
// Declare and initialise an xa_state on the stack.
// xas_for_each() - Iterate over a range of an XArray.
// @xas: XArray operation state.
// @entry: Entry retrieved from the array.
// @max: Maximum index to retrieve from array.
//
// The loop body will be executed for each entry present in the xarray
// between the current xas position and @max.  @entry will be set to
// the entry retrieved from the xarray.  It is safe to delete entries
// from the array in the loop body.  You should hold either the RCU lock
// or the xa_lock while iterating.  If you need to drop the lock, call
// xas_pause() first.
#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru_one {
    list: list_head,
    nr_items: i64,
    lock: spinlock,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct list_lru_node {
    lru: list_lru_one,
    nr_items: __SlateRecord_types_h_197,
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
struct list_lru {
    node: *mut list_lru_node,
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
fn __acquire_ctx_lock__11(var: &__context_bitlock) {
}

#[inline(always)]
fn __release_ctx_lock__11(var: &__context_bitlock) {
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
struct tty_struct {
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
struct tty_audit_buf {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct xattr_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fserror_event {
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
struct cpu_itimer {
    expires: u64,
    incr: u64,
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
    __slate_bits_0: __slate_bits::__SlateBits289U0,
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
struct fiemap_extent_info {
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
struct poll_table_struct {
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
struct posix_acl {
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
struct sb_writers {
    frozen: u16,
    freeze_kcount: i32,
    freeze_ucount: i32,
    freeze_owner: *const (),
    rw_sem: [percpu_rw_semaphore; 3],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
struct __SlateRecord_mnt_idmapping_h_17 {
    val: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits592U0,
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


// XArray tags, for tagging dirty and writeback pages in the pagecache.
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
struct wake_irq {
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
struct dev_pm_qos {
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
struct compat_robust_list {
    next: u32,
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
struct compat_robust_list_head {
    list: compat_robust_list,
    futex_offset: i32,
    list_op_pending: u32,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
    __slate_bits_0: __slate_bits::__SlateBits669U0,
    vm_ops: *const vm_operations_struct,
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
    __slate_bits_0: __slate_bits::__SlateBits605U0,
}


fn kobject_name(kobj: *const kobject) -> *const u8 {
unsafe { (*kobj).name }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
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
struct bio_alloc_cache {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct klist {
    k_lock: spinlock,
    k_list: list_head,
    get: Option<unsafe extern "C" fn(*mut klist_node)>,
    put: Option<unsafe extern "C" fn(*mut klist_node)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct klist_node {
    n_klist: *mut (),
    n_node: list_head,
    n_ref: kref,
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
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
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
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits694U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits694U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits694U2,
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
    __slate_bits_3: __slate_bits::__SlateBits694U3,
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
struct timer_rand_state {
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
struct blk_mq_ctx {
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
struct bsg_device {
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
struct scsi_device_handler {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_attribute {
    attr: attribute,
    __slate_anon_1: __SlateRecord_device_h_108,
    __slate_anon_2: __SlateRecord_device_h_114,
}


// DEVICE_ATTR - Define a device attribute.
// @_name: Attribute name.
// @_mode: File mode.
// @_show: Show handler. Optional, but mandatory if attribute is readable.
// @_store: Store handler. Optional, but mandatory if attribute is writable.
//
// Convenience macro for defining a struct device_attribute.
//
// For example, ``DEVICE_ATTR(foo, 0644, foo_show, foo_store);`` expands to:
//
// .. code-block:: c
//
// 	struct device_attribute dev_attr_foo = {
// 		.attr	= { .name = "foo", .mode = 0644 },
// 		.show	= foo_show,
// 		.store	= foo_store,
// 	};
#[repr(C)]
#[derive(Clone, Copy)]
struct device_dma_parameters {
    max_segment_size: u32,
    min_align_mask: u32,
    segment_boundary_mask: u64,
}


// Device link flags.
//
// STATELESS: The core will not remove this link automatically.
// AUTOREMOVE_CONSUMER: Remove the link automatically on consumer driver unbind.
// PM_RUNTIME: If set, the runtime PM framework will use this link.
// RPM_ACTIVE: Run pm_runtime_get_sync() on the supplier during link creation.
// AUTOREMOVE_SUPPLIER: Remove the link automatically on supplier driver unbind.
// AUTOPROBE_CONSUMER: Probe consumer driver automatically after supplier binds.
// MANAGED: The core tracks presence of supplier/consumer drivers (internal).
// SYNC_STATE_ONLY: Link only affects sync_state() behavior.
// INFERRED: Inferred from data (eg: firmware) and not from driver actions.
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
struct device_link {
    supplier: *mut device,
    s_node: list_head,
    consumer: *mut device,
    c_node: list_head,
    link_dev: device,
    status: i32,
    flags: u32,
    rpm_active: refcount_struct,
    kref: kref,
    rm_work: work_struct,
    supplier_preactivated: bool,
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
    __slate_bits_0: __slate_bits::__SlateBits691U0,
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


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
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


fn blk_op_is_passthrough(mut op: u32) -> bool {
op &= 255u32;
op == 34 || op == 35
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
union __SlateRecord_libata_h_771 {
    id: [u16; 256],
    gscr: [u32; 128],
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

// The Well Known LUNS (SAM-3) in our int representation of a LUN
fn scsi_status_is_check_condition(mut status: i32) -> i32 {
if status < 0 {
return 0;
}
status &= 254;
(status == 2) as i32
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
struct fwnode_endpoint {
    port: u32,
    id: u32,
    local_fwnode: *const fwnode_handle,
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
struct sg_page_iter {
    sg: *mut scatterlist,
    sg_pgoffset: u32,
    __nents: u32,
    __pg_advance: i32,
}


// use kmap_atomic
// flush back to phys on unmap
#[repr(C)]
#[derive(Clone, Copy)]
struct sg_mapping_iter {
    page: *mut page,
    addr: *mut (),
    length: u64,
    consumed: u64,
    piter: sg_page_iter,
    __offset: u32,
    __remaining: u32,
    __flags: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_taskset {
}


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


fn blk_rq_is_passthrough(rq: *mut request) -> bool {
blk_op_is_passthrough(unsafe { (*rq).cmd_flags })
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
struct swap_cluster_info {
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
struct swap_ops {
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
struct __SlateRecord_libata_scsi_c_189 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libata_scsi_c_189_2 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libata_scsi_c_189_3 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libata_scsi_c_189_4 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libata_scsi_c_189_5 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libata_scsi_c_189_6 {
}


fn blk_mq_rq_from_pdu(pdu: *mut ()) -> *mut request {
((unsafe { (pdu as *const u8).wrapping_offset(-(core::mem::size_of::<request>() as isize)) }) as *mut ()) as *mut request
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libata_scsi_c_3863 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libata_scsi_c_3877 {
}


fn blk_mq_rq_to_pdu(rq: *mut request) -> *mut () {
rq.wrapping_offset(1 as isize) as *mut ()
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_libata_scsi_c_3959 {
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
    pub struct __SlateBits560U0 {
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
    pub struct __SlateBits592U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits286U0 {
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
    pub struct __SlateBits77U0 {
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
    pub struct __SlateBits77U1 {
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
    pub struct __SlateBits669U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits939U0 {
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
    pub struct __SlateBits289U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits694U0 {
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
    pub struct __SlateBits691U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits694U1 {
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
    pub struct __SlateBits694U2 {
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
    pub struct __SlateBits694U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits801U0 {
        #[bits(30)]
        pub options: u32,
        #[bits(2)]
        pub mc_flags: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits801U1 {
        #[bits(2)]
        pub sanyo_slot: u8,
        #[bits(1)]
        pub keeplocked: u8,
        #[bits(5)]
        pub reserved: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits77U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits124U0 {
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
    pub struct __SlateBits130U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits605U0 {
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
    pub struct __SlateBits129U0 {
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
    pub struct __SlateBits416U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits840U0 {
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
    #[bitfields::bitfield([u8; 8], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits839U0 {
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
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits844U0 {
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
    pub struct __SlateBits841U0 {
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
    pub struct __SlateBits842U0 {
        #[bits(1)]
        pub create_work_queue: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
}

fn blk_rq_bytes(rq: *const request) -> u32 {
unsafe { (*rq).__data_len }
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
    __slate_bits_0: __slate_bits::__SlateBits844U0,
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
    __slate_bits_0: __slate_bits::__SlateBits841U0,
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
    __slate_bits_0: __slate_bits::__SlateBits839U0,
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
    __slate_bits_0: __slate_bits::__SlateBits840U0,
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

fn scsi_sg_count(cmd: *mut scsi_cmnd) -> u32 {
unsafe { (*cmd).sdb.table.nents }
}

fn scsi_sglist(cmd: *mut scsi_cmnd) -> *mut scatterlist {
unsafe { (*cmd).sdb.table.sgl }
}

fn scsi_bufflen(cmd: *mut scsi_cmnd) -> u32 {
unsafe { (*cmd).sdb.length }
}

fn scsi_set_resid(cmd: *mut scsi_cmnd, resid: u32) {
unsafe {
(*cmd).resid_len = resid;
}
}

fn set_status_byte(cmd: *mut scsi_cmnd, status: u8) {
unsafe {
(*cmd).result = (((*cmd).result as u32) & 4294967040 | ((status as i32) as u32)) as i32;
}
}

fn set_host_byte(cmd: *mut scsi_cmnd, status: u8) {
unsafe {
(*cmd).result = (((*cmd).result as u32) & 4278255615 | (((status as i32) << 16i32) as u32)) as i32;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct attribute_container {
    node: list_head,
    containers: klist,
    class: *mut class,
    grp: *const attribute_group,
    attrs: *mut *mut device_attribute,
    r#match: Option<unsafe extern "C" fn(*mut attribute_container, *mut device) -> i32>,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct transport_container {
    ac: attribute_container,
    statistics: *const attribute_group,
    encryption: *const attribute_group,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_transport_template {
    host_attrs: transport_container,
    target_attrs: transport_container,
    device_attrs: transport_container,
    user_scan: Option<unsafe extern "C" fn(*mut Scsi_Host, u32, u32, u64) -> i32>,
    device_size: i32,
    device_private_offset: i32,
    target_size: i32,
    target_private_offset: i32,
    host_size: i32,
    __slate_bits_0: __slate_bits::__SlateBits842U0,
    eh_strategy_handler: Option<unsafe extern "C" fn(*mut Scsi_Host)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ata_bmdma_prd {
    addr: u32,
    flags_len: u32,
}


fn ata_id_has_flush(id: *const u16) -> bool {
if ((unsafe { *id.wrapping_offset(83 as isize) }) as i32) & 49152 != 16384 {
return false;
}
((unsafe { *id.wrapping_offset(83 as isize) }) as i32) & 4096i32 != 0
}

fn ata_id_has_flush_ext(id: *const u16) -> bool {
if ((unsafe { *id.wrapping_offset(83 as isize) }) as i32) & 49152 != 16384 {
return false;
}
((unsafe { *id.wrapping_offset(83 as isize) }) as i32) & 8192i32 != 0
}

fn ata_id_logical_sector_size(id: *const u16) -> u32 {
// T13/1699-D Revision 6a, Sep 6, 2008. Page 128.
// IDENTIFY DEVICE data, word 117-118.
// 0xd000 ignores bit 13 (logical:physical > 1)
if ((unsafe { *id.wrapping_offset(106 as isize) }) as i32) & 53248 == 20480 {
return ((((unsafe { *id.wrapping_offset(118i32 as isize) }) as i32) << 16i32).wrapping_add((unsafe { *id.wrapping_offset(117 as isize) }) as i32) as u64).wrapping_mul(core::mem::size_of::<u16>() as u64) as u32;
}
512
}

fn ata_id_log2_per_physical_sector(id: *const u16) -> u8 {
// T13/1699-D Revision 6a, Sep 6, 2008. Page 128.
// IDENTIFY DEVICE data, word 106.
// 0xe000 ignores bit 12 (logical sector > 512 bytes)
if ((unsafe { *id.wrapping_offset(106 as isize) }) as i32) & 57344 == 24576 {
return (((unsafe { *id.wrapping_offset(106 as isize) }) as i32) & 15) as u8;
}
0
}

fn ata_id_logical_sector_offset(id: *const u16, log2_per_phys: u8) -> u16 {
let word_209: u16 = unsafe { *id.wrapping_offset(209 as isize) };
if (log2_per_phys as i32) > 1 && (word_209 as i32) & 49152 == 16384 {
let first: u16 = ((word_209 as i32) & 16383) as u16;
if (first as i32) > 0 {
return (1i32 << (log2_per_phys as i32)).wrapping_sub(first as i32) as u16;
}
}
0
}

fn ata_id_rahead_enabled(id: *const u16) -> bool {
if ((unsafe { *id.wrapping_offset(87 as isize) }) as i32) & 49152 != 16384 {
return false;
}
((unsafe { *id.wrapping_offset(85 as isize) }) as i32) & 64i32 != 0
}

fn ata_id_wcache_enabled(id: *const u16) -> bool {
if ((unsafe { *id.wrapping_offset(87 as isize) }) as i32) & 49152 != 16384 {
return false;
}
((unsafe { *id.wrapping_offset(85 as isize) }) as i32) & 32i32 != 0
}

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_dev_attr_unload_heads: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_common_sdev_groups: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_std_bios_param: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_unlock_native_capacity: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_sas_scsi_ioctl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_ioctl: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_dma_need_drain: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_sdev_init: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_sdev_configure: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_sdev_destroy: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_retry_deferred_qc: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_eh_timed_out: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ata_scsi_queuecmd: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous", sym dev_attr_unload_heads, sym ata_common_sdev_groups, sym ata_std_bios_param, sym ata_scsi_unlock_native_capacity, sym ata_sas_scsi_ioctl, sym ata_scsi_ioctl, sym ata_scsi_dma_need_drain, sym ata_scsi_sdev_init, sym ata_scsi_sdev_configure, sym ata_scsi_sdev_destroy, sym ata_scsi_retry_deferred_qc, sym ata_scsi_eh_timed_out, sym ata_scsi_queuecmd, options(att_syntax));

fn ata_id_major_version(id: *const u16) -> u32 {
let mut mver: u32 = 14;
if ((unsafe { *id.wrapping_offset(80 as isize) }) as i32) == 65535 {
return 0;
}
while mver >= 1 {
if ((unsafe { *id.wrapping_offset(80 as isize) }) as i32) & 1i32 << mver != 0 {
break;
}
mver = mver.wrapping_sub(1);
}
mver
}

fn ata_id_has_unload(id: *const u16) -> bool {
if ata_id_major_version(id) >= 7 && ((unsafe { *id.wrapping_offset(84 as isize) }) as i32) & 49152 == 16384 && ((unsafe { *id.wrapping_offset(84 as isize) }) as i32) & 8192i32 != 0 {
return true;
}
false
}

fn ata_id_has_wwn(id: *const u16) -> bool {
((unsafe { *id.wrapping_offset(87 as isize) }) as i32) & 49408 == 16640
}

fn ata_id_form_factor(id: *const u16) -> i32 {
let mut val: u16 = unsafe { *id.wrapping_offset(168 as isize) };
if ata_id_major_version(id) < 7 || (val as i32) == 0 || (val as i32) == 65535 {
return 0;
}
val = ((val as i32) & 15) as u16;
if (val as i32) > 5 {
return 0;
}
val as i32
}

fn ata_id_rotation_rate(id: *const u16) -> i32 {
let val: u16 = unsafe { *id.wrapping_offset(217 as isize) };
if ata_id_major_version(id) < 7 || (val as i32) == 0 || (val as i32) == 65535 {
return 0;
}
if (val as i32) > 1 && (val as i32) < 1025 {
return 0;
}
val as i32
}

fn ata_id_has_trim(id: *const u16) -> bool {
if ata_id_major_version(id) >= 7 && ((unsafe { *id.wrapping_offset(169 as isize) }) as i32) & 1 != 0 {
return true;
}
false
}

fn ata_id_dsm_max_pages(id: *const u16) -> u16 {
// IDENTIFY DEVICE word 105: MAX PAGES PER DSM COMMAND. Maximum number
// of 512-byte pages of LBA Range Entries the device accepts in a
// single DATA SET MANAGEMENT command. Zero means the device does not
// specify a limit. The field is reserved unless TRIM is supported, so
// callers must gate on ata_id_has_trim().
unsafe { *id.wrapping_offset(105 as isize) }
}

fn ata_id_has_zero_after_trim(id: *const u16) -> bool {
// DSM supported, deterministic read, and read zero after trim set
if ata_id_has_trim(id) && ((unsafe { *id.wrapping_offset(69 as isize) }) as i32) & 16416 == 16416 {
return true;
}
false
}

fn ata_id_zoned_cap(id: *const u16) -> u8 {
(((unsafe { *id.wrapping_offset(69 as isize) }) as i32) & 3) as u8
}

fn lba_28_ok(block: u64, n_block: u32) -> bool {
// check the ending block number: must be LESS THAN 0x0fffffff
block.wrapping_add(n_block as u64) < 268435455u64 && n_block <= 256
}

fn lba_48_ok(block: u64, n_block: u32) -> bool {
// check the ending block number
block.wrapping_add(n_block as u64).wrapping_sub(1) < 281474976710656u64 && n_block <= 65535
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
    __slate_bits_0: __slate_bits::__SlateBits801U0,
    vfs_events: u32,
    ioctl_events: u32,
    use_count: i32,
    name: [u8; 20],
    __slate_bits_1: __slate_bits::__SlateBits801U1,
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
#[unsafe(no_mangle)]
static mut dev_attr_unload_heads: device_attribute = device_attribute { attr: {
let mut __t0: attribute = unsafe { core::mem::zeroed() };
__t0.name = (b"unload_heads\0".as_ptr() as *mut u8) as *const u8;
__t0.mode = ((core::mem::size_of::<__SlateRecord_libata_scsi_c_189>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libata_scsi_c_189_2>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libata_scsi_c_189_3>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libata_scsi_c_189_4>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libata_scsi_c_189_5>() as u64) as i32).wrapping_add((core::mem::size_of::<__SlateRecord_libata_scsi_c_189_6>() as u64) as i32).wrapping_add(420i32) as u16;
__t0
}, __slate_anon_1: __SlateRecord_device_h_108 { show: Some(ata_scsi_park_show) }, __slate_anon_2: __SlateRecord_device_h_114 { store: Some(ata_scsi_park_store) } };

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


fn ata_is_ncq(prot: u8) -> bool {
(prot as i32) & 4 != 0
}

fn is_multi_taskfile(tf: *mut ata_taskfile) -> i32 {
(((unsafe { (*tf).__slate_anon_14.command }) as i32) == 196 || ((unsafe { (*tf).__slate_anon_14.command }) as i32) == 197 || ((unsafe { (*tf).__slate_anon_14.command }) as i32) == 41 || ((unsafe { (*tf).__slate_anon_14.command }) as i32) == 57 || ((unsafe { (*tf).__slate_anon_14.command }) as i32) == 206) as i32
}

fn ata_port_is_frozen(ap: *const ata_port) -> bool {
(unsafe { (*ap).pflags }) & 4 != 0
}

fn sata_pmp_attached(ap: *mut ata_port) -> bool {
(unsafe { (*ap).nr_pmp_links }) != 0
}

fn ata_is_host_link(link: *const ata_link) -> bool {
link == ((unsafe { core::ptr::addr_of_mut!((*(*link).ap).link) }) as *const ata_link) || link == ((unsafe { (*(*link).ap).slave_link }) as *const ata_link)
}

fn ata_tag_internal(tag: u32) -> bool {
tag == 32
}

fn ata_tag_valid(tag: u32) -> bool {
tag < 32 || ata_tag_internal(tag)
}

fn ata_class_enabled(class: u32) -> u32 {
(class == 1 || class == 3 || class == 5 || class == 7 || class == 9) as u32
}

fn ata_class_disabled(class: u32) -> u32 {
(class == 2 || class == 4 || class == 6 || class == 8 || class == 10) as u32
}

fn ata_dev_enabled(dev: *const ata_device) -> u32 {
ata_class_enabled(unsafe { (*dev).class })
}

fn ata_dev_disabled(dev: *const ata_device) -> u32 {
ata_class_disabled(unsafe { (*dev).class })
}

fn ata_link_max_devices(link: *const ata_link) -> i32 {
if ata_is_host_link(link) && (unsafe { (*(*link).ap).flags }) & 1 != 0 {
return 2;
}
1
}

// Shortcut notation for iterations
//
// ata_for_each_link() iterates over each link of @ap according to
// @mode.  @link points to the current link in the loop.  @link is
// NULL after loop termination.  ata_for_each_dev() works the same way
// except that it iterates over each device of @link.
//
// Note that the mode prefixes ATA_{L|D}ITER_ shouldn't need to be
// specified when using the following shorthand notations.  Only the
// mode itself (EDGE, HOST_FIRST, ENABLED, etc...) should be
// specified.  This not only increases brevity but also makes it
// impossible to use ATA_LITER_* for device iteration or vice-versa.
fn ata_ncq_supported(dev: *mut ata_device) -> bool {
if IS_ENABLED__2!() == 0 {
return false;
}
(unsafe { (*dev).flags }) & 131080u64 == 8
}

fn ata_ncq_enabled(dev: *mut ata_device) -> bool {
ata_ncq_supported(dev) && (unsafe { (*dev).flags }) & 262144 == 0
}

fn ata_fpdma_dsm_supported(dev: *mut ata_device) -> bool {
(unsafe { (*dev).flags }) & 2048 != 0 && ((unsafe { *((*dev).ncq_send_recv_cmds.as_mut_ptr() as *mut u8).wrapping_offset(4 as isize) }) as i32) & 1 != 0
}

fn ata_fpdma_zac_mgmt_in_supported(dev: *mut ata_device) -> bool {
(unsafe { (*dev).flags }) & 2048 != 0 && ((unsafe { *((*dev).ncq_send_recv_cmds.as_mut_ptr() as *mut u8).wrapping_offset(16 as isize) }) as i32) & 2 != 0
}

fn ata_fpdma_zac_mgmt_out_supported(dev: *mut ata_device) -> bool {
((unsafe { *((*dev).ncq_non_data_cmds.as_mut_ptr() as *mut u8).wrapping_offset(28 as isize) }) as i32) & 1 != 0
}

fn __ata_qc_from_tag(ap: *mut ata_port, tag: u32) -> *mut ata_queued_cmd {
if ata_tag_valid(tag) {
return unsafe { (*ap).qcmd.as_mut_ptr() as *mut ata_queued_cmd }.wrapping_offset(tag as isize);
}
core::ptr::null_mut::<ata_queued_cmd>()
}

fn ata_qc_raw_nbytes(qc: *mut ata_queued_cmd) -> u32 {
let mut __v0: u32;
let __UNIQUE_ID_x__599: u32 = unsafe { (*qc).extrabytes };
let __UNIQUE_ID_y__600: u32 = unsafe { (*qc).nbytes };
loop {
if __types_ok__2!(__UNIQUE_ID_x__599, __UNIQUE_ID_y__600) == 0 {
unsafe { __compiletime_assert_601() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__599 < __UNIQUE_ID_y__600 { __UNIQUE_ID_x__599 } else { __UNIQUE_ID_y__600 };
unsafe { (*qc).nbytes }.wrapping_sub(__v0)
}

fn ata_tf_init(dev: *mut ata_device, tf: *mut ata_taskfile) {
unsafe { memset(tf as *mut (), 0i32, core::mem::size_of::<ata_taskfile>() as u64) };
unsafe {
(*tf).ctl = (*(*(*dev).link).ap).ctl;
}
if (unsafe { (*dev).devno }) == 0 {
unsafe {
(*tf).device = 160;
}
} else {
unsafe {
(*tf).device = 176u8;
}
}
}

fn ata_qc_reinit(qc: *mut ata_queued_cmd) {
unsafe {
(*qc).dma_dir = 3;
}
unsafe {
(*qc).sg = core::ptr::null_mut::<scatterlist>();
}
unsafe {
(*qc).flags = 0;
}
unsafe {
(*qc).cursg = core::ptr::null_mut::<scatterlist>();
}
unsafe {
(*qc).cursg_ofs = 0;
}
unsafe {
(*qc).curbytes = 0;
}
unsafe {
(*qc).extrabytes = 0;
}
unsafe {
(*qc).nbytes = 0;
}
unsafe {
(*qc).n_elem = 0;
}
unsafe {
(*qc).err_mask = 0;
}
unsafe {
(*qc).sect_size = 512;
}
ata_tf_init(unsafe { (*qc).dev }, unsafe { core::ptr::addr_of_mut!((*qc).tf) });
// init result_tf such that it indicates normal completion
unsafe {
(*qc).result_tf.__slate_anon_14.command = 64;
}
unsafe {
(*qc).result_tf.__slate_anon_8.feature = 0;
}
}

fn ata_try_flush_cache(dev: *const ata_device) -> i32 {
let __v0: bool = ata_id_wcache_enabled(unsafe { (*dev).__slate_anon_26.id.as_ptr() as *const u16 }) || ata_id_has_flush(unsafe { (*dev).__slate_anon_26.id.as_ptr() as *const u16 });
let __v1: bool = __v0 || ata_id_has_flush_ext(unsafe { (*dev).__slate_anon_26.id.as_ptr() as *const u16 });
__v1 as i32
}

fn ata_shost_to_port(host: *mut Scsi_Host) -> *mut ata_port {
unsafe { *((core::ptr::addr_of_mut!((*host).hostdata) as *mut u64).wrapping_offset(0 as isize) as *mut *mut ata_port) }
}

fn ata_deadline(from_jiffies: u64, timeout_msecs: u32) -> u64 {
from_jiffies.wrapping_add(msecs_to_jiffies(timeout_msecs))
}

fn ata_dma_enabled(adev: *mut ata_device) -> bool {
((unsafe { (*adev).dma_mode }) as i32) != 255
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hd_geometry {
    heads: u8,
    sectors: u8,
    cylinders: u16,
    start: u64,
}


// get current io_32bit setting
// get IDE identification info
// execute task and special drive command
// execute a special drive command
// change io_32bit flags
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
    __slate_bits_0: __slate_bits::__SlateBits939U0,
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
    __slate_bits_0: __slate_bits::__SlateBits560U0,
    fbatch: folio_batch,
    index: u64,
    saved_err: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct wait_page_queue {
    folio: *mut folio,
    bit_nr: i32,
    wait: wait_queue_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct readahead_control {
    file: *mut file,
    mapping: *mut address_space,
    ra: *mut file_ra_state,
    _index: u64,
    _nr_pages: u32,
    _batch_count: u32,
    dropbehind: bool,
    _workingset: bool,
    _pflags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct reclaim_state {
    reclaimed: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_sequential_cluster {
    next: [u32; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
    users: percpu_ref,
    flags: u64,
    prio: i16,
    list: plist_node,
    r#type: i8,
    max: u32,
    cluster_info: *mut swap_cluster_info,
    free_clusters: list_head,
    full_clusters: list_head,
    nonfull_clusters: [list_head; 1],
    frag_clusters: [list_head; 1],
    pages: u32,
    inuse_pages: __SlateRecord_types_h_197,
    global_cluster: *mut swap_sequential_cluster,
    global_cluster_lock: spinlock,
    swap_extent_root: rb_root,
    bdev: *mut block_device,
    swap_file: *mut file,
    comp: completion,
    lock: spinlock,
    discard_work: work_struct,
    reclaim_work: work_struct,
    discard_clusters: list_head,
    avail_list: plist_node,
    ops: *const swap_ops,
}


fn get_unaligned_le16(p: *const ()) -> u16 {
__get_unaligned_t!(u16, p)
}

fn get_unaligned_le32(p: *const ()) -> u32 {
__get_unaligned_t__2!(u32, p)
}

fn get_unaligned_le64(p: *const ()) -> u64 {
__get_unaligned_t__3!(u64, p)
}

fn get_unaligned_be16(p: *const ()) -> u16 {
__swab16!(__get_unaligned_t!(u16, p))
}

fn get_unaligned_be32(p: *const ()) -> u32 {
__swab32!(__get_unaligned_t__2!(u32, p))
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

fn __get_unaligned_be24(p: *const u8) -> u32 {
(((unsafe { *p.wrapping_offset(0 as isize) }) as i32) << 16i32 | ((unsafe { *p.wrapping_offset(1 as isize) }) as i32) << 8i32 | ((unsafe { *p.wrapping_offset(2 as isize) }) as i32)) as u32
}

fn get_unaligned_be24(p: *const ()) -> u32 {
__get_unaligned_be24(p as *const u8)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct property {
    name: *mut u8,
    length: i32,
    value: *mut (),
    next: *mut property,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct device_node {
    name: *const u8,
    phandle: u32,
    full_name: *const u8,
    fwnode: fwnode_handle,
    properties: *mut property,
    deadprops: *mut property,
    parent: *mut device_node,
    child: *mut device_node,
    sibling: *mut device_node,
    _flags: u64,
    data: *mut (),
}


fn ata_dev_is_zoned(dev: *mut ata_device) -> bool {
// Host managed device or host aware device
(unsafe { (*dev).class }) == 9 || (ata_id_zoned_cap((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16) as i32) == 1
}

fn ata_port_eh_scheduled(ap: *mut ata_port) -> bool {
(unsafe { (*ap).pflags }) & 3u32 != 0
}

// libata-scsi.c
#[unsafe(no_mangle)]
extern "C" fn ata_scsi_dev_config(sdev: *mut scsi_device, lim: *mut queue_limits, dev: *mut ata_device) -> i32 {
let mut __slate_compound_10: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut depth: i32 = 1;
if !ata_id_has_unload((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16) {
unsafe {
(*dev).flags = (*dev).flags | 2097152;
}
}
// configure max sectors
let mut __v3: u32;
let __UNIQUE_ID_x__683: u32 = unsafe { (*dev).max_sectors };
let __UNIQUE_ID_y__684: u32 = unsafe { (*(*sdev).host).max_sectors };
loop {
if __types_ok__2!(__UNIQUE_ID_x__683, __UNIQUE_ID_y__684) == 0 {
unsafe { __compiletime_assert_685() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__683 < __UNIQUE_ID_y__684 { __UNIQUE_ID_x__683 } else { __UNIQUE_ID_y__684 };
unsafe {
(*dev).max_sectors = __v3;
}
unsafe {
(*lim).max_hw_sectors = (*dev).max_sectors;
}
if (unsafe { (*dev).class }) == 3 {
unsafe {
(*sdev).sector_size = 512;
}
// set DMA padding
unsafe {
(*lim).dma_pad_mask = 3u32;
}
// make room for appending the drain
unsafe {
(*lim).max_segments = ((*lim).max_segments as i32).wrapping_sub(1) as u16;
}
unsafe {
(*sdev).dma_drain_len = 16384;
}
let mut __v8: *mut ();
let mut __v9: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(unsafe { (*sdev).dma_drain_len }, 3072u32, unsafe { *{
__slate_compound_10 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_10)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(unsafe { (*sdev).dma_drain_len }, 3072u32, unsafe { *{
__slate_compound_12 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_12)
} });
}
__v9 = _res;
__v8 = __v9;
unsafe {
(*sdev).dma_drain_buf = __v8;
}
if unsafe { (*sdev).dma_drain_buf }.is_null() {
let mut __v13: i32;
while false {
}
__v13 = unsafe { _printk(c"\x013ata%u.%02u: drain buffer allocation failed\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
return 12i32.wrapping_neg();
}
} else {
unsafe {
(*sdev).sector_size = ata_id_logical_sector_size(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16);
}
// Ask the sd driver to issue START STOP UNIT on runtime suspend
// and resume and shutdown only. For system level suspend/resume,
// devices power state is handled directly by libata EH.
// Given that disks are always spun up on system resume, also
// make sure that the sd driver forces runtime suspended disks
// to be resumed to correctly reflect the power state of the
// device.
unsafe {
(*sdev).__slate_bits_0.__set_manage_runtime_start_stop(1u32);
}
unsafe {
(*sdev).__slate_bits_0.__set_manage_shutdown(1u32);
}
unsafe {
let __t0 = ata_acpi_dev_manage_restart(dev) as u32;
(*sdev).__slate_bits_0.__set_manage_restart(__t0);
}
unsafe {
(*sdev).__slate_bits_0.__set_force_runtime_start_on_system_start(1u32);
}
}
// ata_pio_sectors() expects buffer for each sector to not cross
// page boundary.  Enforce it by requiring buffers to be sector
// aligned, which works iff sector_size is not larger than
// PAGE_SIZE.  ATAPI devices also need the alignment as
// IDENTIFY_PACKET is executed as ATA_PROT_PIO.
if ((unsafe { (*sdev).sector_size }) as u64) > _AC!() << 12i32 {
let mut __v15: i32;
while false {
}
__v15 = unsafe { _printk(c"\x014ata%u.%02u: sector_size=%u > PAGE_SIZE, PIO may malfunction\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno), (*sdev).sector_size) };
}
unsafe {
(*lim).dma_alignment = unsafe { (*sdev).sector_size }.wrapping_sub(1);
}
if (unsafe { (*dev).flags }) & 128 != 0 {
set_bit(1i64, (unsafe { (*sdev).supported_events.as_mut_ptr() as *mut u64 }) as *mut u64);
}
if ata_ncq_supported(dev) {
let mut __v17: i32;
let __UNIQUE_ID_x__686: i32 = unsafe { (*(*sdev).host).can_queue };
let __UNIQUE_ID_y__687: i32 = ata_id_queue_depth!((*dev).__slate_anon_26.id);
loop {
if __types_ok!(__UNIQUE_ID_x__686, __UNIQUE_ID_y__687) == 0 {
unsafe { __compiletime_assert_688() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v17 = if __UNIQUE_ID_x__686 < __UNIQUE_ID_y__687 { __UNIQUE_ID_x__686 } else { __UNIQUE_ID_y__687 };
depth = __v17;
}
let mut __v19: i32;
let __UNIQUE_ID_x__689: i32 = 32;
let __UNIQUE_ID_y__690: i32 = depth;
loop {
if __types_ok!(__UNIQUE_ID_x__689, __UNIQUE_ID_y__690) == 0 {
unsafe { __compiletime_assert_691() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v19 = if __UNIQUE_ID_x__689 < __UNIQUE_ID_y__690 { __UNIQUE_ID_x__689 } else { __UNIQUE_ID_y__690 };
depth = __v19;
unsafe { scsi_change_queue_depth(sdev, depth) };
if (unsafe { (*dev).flags }) & 256 != 0 {
unsafe {
(*sdev).__slate_bits_0.__set_security_supported(1u32);
}
}
unsafe {
(*dev).sdev = sdev;
}
0
}

static mut ata_scsi_rbuf_lock: spinlock = spinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_spinlock_types_h_18 = unsafe { core::mem::zeroed() };
__t0.rlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t1: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t1.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t1
} } };
__t0
} };

static mut ata_scsi_rbuf: __SlateAlign16<[u8; 2048]> = __SlateAlign16(unsafe { core::mem::zeroed() });

/// AWRE
/// read retry count
/// write retry count
static def_rw_recovery_mpage: [u8; 12] = [1, 10, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0];

/// contains WCE, needs to be 0 for logic
/// contains DRA, needs to be 0 for logic
static def_cache_mpage: __SlateAlign16<[u8; 20]> = __SlateAlign16([8u8, 18u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8]);

/// DSENSE=0, GLTSD=1
/// [QAM+QERR may be 1, see 05-359r1]
/// extended self test time, see 05-359r1
static def_control_mpage: [u8; 12] = [10, 10, 2, 0, 0, 0, 0, 0, 255, 255, 0, 30];

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_park_show")]
extern "C" fn ata_scsi_park_show(device: *mut device, attr: *mut device_attribute, buf: *mut u8) -> i64 {
let mut __v0: i32 = unsafe { core::mem::zeroed() };
let mut __v1: bool = unsafe { core::mem::zeroed() };
let mut rc: i32 = unsafe { core::mem::zeroed() };
let mut msecs: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u32 = core::ptr::addr_of_mut!(msecs) as *mut u32;
let mut now: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u64 = core::ptr::addr_of_mut!(now) as *mut u64;
let mut dev: core::mem::MaybeUninit<*mut ata_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut ata_device = core::ptr::addr_of_mut!(dev) as *mut *mut ata_device;
let mut link: core::mem::MaybeUninit<*mut ata_link> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut ata_link = core::ptr::addr_of_mut!(link) as *mut *mut ata_link;
let mut ap: core::mem::MaybeUninit<*mut ata_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ata_port = core::ptr::addr_of_mut!(ap) as *mut *mut ata_port;
unsafe {
let sdev: *mut scsi_device = to_scsi_device!(device);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
rc = 0;
*__slate_slot_7 = ata_shost_to_port((*sdev).host);
spin_lock_irq((*(*__slate_slot_7)).lock);
*__slate_slot_5 = ata_scsi_find_dev(*__slate_slot_7, sdev as *const scsi_device);
if (*__slate_slot_5).is_null() {
rc = 19i32.wrapping_neg();
} else {
if (*(*__slate_slot_5)).flags & 2097152 != 0 {
rc = 95i32.wrapping_neg();
} else {
*__slate_slot_6 = (*(*__slate_slot_5)).link;
*__slate_slot_4 = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
if (*(*__slate_slot_7)).pflags & 2 != 0 && (*(*__slate_slot_6)).eh_context.unloaded_mask & ((1i32 << (*(*__slate_slot_5)).devno) as u32) != 0 {
__v1 = time_after__2!(unsafe { (*(*__slate_slot_5)).unpark_deadline }, *__slate_slot_4);
} else {
__v1 = false;
}
if __v1 {
*__slate_slot_3 = jiffies_to_msecs(unsafe { (*(*__slate_slot_5)).unpark_deadline }.wrapping_sub(*__slate_slot_4));
} else {
*__slate_slot_3 = 0;
}
}
}
spin_unlock_irq((*(*__slate_slot_7)).lock);
if rc != 0 {
__v0 = rc;
} else {
__v0 = sysfs_emit(buf, c"%u\n".as_ptr() as *const u8, *__slate_slot_3);
}
return __v0 as i64;
}
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_park_store")]
extern "C" fn ata_scsi_park_store(device: *mut device, attr: *mut device_attribute, buf: *const u8, len: u64) -> i64 {
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut input: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(input) as *mut i32;
let mut dev: core::mem::MaybeUninit<*mut ata_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut ata_device = core::ptr::addr_of_mut!(dev) as *mut *mut ata_device;
let mut ap: core::mem::MaybeUninit<*mut ata_port> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut ata_port = core::ptr::addr_of_mut!(ap) as *mut *mut ata_port;
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
unsafe {
sdev = to_scsi_device!(device);
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
*__slate_slot_9 = kstrtoint(buf, 10u32, core::ptr::addr_of_mut!(*__slate_slot_11));
if *__slate_slot_9 != 0 {
return *__slate_slot_9 as i64;
} else {
if *__slate_slot_11 < 2i32.wrapping_neg() {
return 22i32.wrapping_neg() as i64;
} else {
if *__slate_slot_11 > 30000 {
*__slate_slot_9 = 75i32.wrapping_neg();
*__slate_slot_11 = 30000;
}
*__slate_slot_13 = ata_shost_to_port((*sdev).host);
spin_lock_irqsave!(unsafe { (*(*__slate_slot_13)).lock }, *__slate_slot_10);
*__slate_slot_12 = ata_scsi_find_dev(*__slate_slot_13, sdev as *const scsi_device);
if unlikely!((*__slate_slot_12).is_null()) != 0 {
*__slate_slot_9 = 19i32.wrapping_neg();
} else {
if (*(*__slate_slot_12)).class != 1 && (*(*__slate_slot_12)).class != 9 {
*__slate_slot_9 = 95i32.wrapping_neg();
} else {
if *__slate_slot_11 >= 0 {
if (*(*__slate_slot_12)).flags & 2097152 != 0 {
*__slate_slot_9 = 95i32.wrapping_neg();
} else {
(*(*__slate_slot_12)).unpark_deadline = ata_deadline(core::ptr::read_volatile(core::ptr::addr_of!(jiffies)), *__slate_slot_11 as u32);
let __v2: *mut u32 = ((*(*(*__slate_slot_12)).link).eh_info.dev_action.as_mut_ptr() as *mut u32).wrapping_offset((*(*__slate_slot_12)).devno as isize);
*__v2 = *__v2 | 32;
ata_port_schedule_eh(*__slate_slot_13);
complete(core::ptr::addr_of_mut!((*(*__slate_slot_13)).park_req_pending));
}
} else {
let __t0: i32 = *__slate_slot_11;
if __t0 == -1 {
let __v8: *mut ata_device = *__slate_slot_12;
let __v6: u64 = (*__v8).flags & 4292870143u64;
(*__v8).flags = __v6;
} else {
if __t0 == -2 {
let __v5: *mut ata_device = *__slate_slot_12;
(*__v5).flags = (*__v5).flags | 2097152;
}
}
}
}
}
spin_unlock_irqrestore((*(*__slate_slot_13)).lock, *__slate_slot_10);
return (if *__slate_slot_9 != 0 { *__slate_slot_9 as u64 } else { len }) as i64;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_dev_attr_unload_heads_674: *mut () = (unsafe { core::ptr::addr_of_mut!(dev_attr_unload_heads) }) as *mut ();

#[unsafe(no_mangle)]
extern "C" fn ata_scsi_sense_is_valid(sk: u8, asc: u8, ascq: u8) -> bool {
// If sk == NO_SENSE, and asc + ascq == NO ADDITIONAL SENSE INFORMATION,
// then there is no sense data to add.
if (sk as i32) == 0 && (asc as i32) == 0 && (ascq as i32) == 0 {
return false;
}
// If sk > COMPLETED, sense data is bogus.
if (sk as i32) > 15 {
return false;
}
true
}

#[unsafe(no_mangle)]
extern "C" fn ata_scsi_set_sense(dev: *mut ata_device, cmd: *mut scsi_cmnd, sk: u8, asc: u8, ascq: u8) {
let d_sense: bool = (unsafe { (*dev).flags }) & 2147483648 != 0;
unsafe { scsi_build_sense(cmd, d_sense as i32, sk, asc, ascq) };
}

fn ata_scsi_set_sense_information(qc: *mut ata_queued_cmd) {
let mut information: u64 = 0;
if (unsafe { (*qc).flags }) & 4 == 0 {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017ata%u.%02u: missing result TF: can't set INFORMATION sense field\n".as_ptr() as *const u8, (*(*(*(*qc).dev).link).ap).print_id, ((*(*(*qc).dev).link).pmp as u32).wrapping_add((*(*qc).dev).devno)) };
}
__v0 = 0;
return;
}
information = unsafe { ata_tf_read_block(core::ptr::addr_of_mut!((*qc).result_tf) as *const ata_taskfile, (*qc).dev) };
if information == !0 {
return;
}
unsafe { scsi_set_sense_information((*(*qc).scsicmd).sense_buffer, 96i32, information) };
}

/// ata_scsi_set_passthru_sense_fields - Set ATA fields in sense buffer
/// @qc: ATA PASS-THROUGH command.
///
/// Populates "ATA Status Return sense data descriptor" / "Fixed format
/// sense data" with ATA taskfile fields.
///
/// LOCKING:
/// None.
fn ata_scsi_set_passthru_sense_fields(qc: *mut ata_queued_cmd) {
let dev: *mut ata_device = unsafe { (*qc).dev };
let cmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).result_tf) };
let sb: *mut u8 = unsafe { (*cmd).sense_buffer };
if (unsafe { (*qc).flags }) & 4 == 0 {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017ata%u.%02u: missing result TF: can't set ATA PT sense fields\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
}
__v0 = 0;
return;
}
if ((unsafe { *sb.wrapping_offset(0 as isize) }) as i32) & 127 >= 114 {
let mut desc: *mut u8 = unsafe { core::mem::zeroed() };
let mut len: u8 = 0;
// descriptor format
len = unsafe { *sb.wrapping_offset(7 as isize) };
desc = (unsafe { scsi_sense_desc_find(sb as *const u8, 96i32, 9i32) }) as *mut u8;
if desc.is_null() {
// The descriptor is written at sb[8 + len] and is 14
// bytes long, so it needs len + 22 bytes of buffer.
if (len as i32).wrapping_add(22) > 96 {
return;
}
unsafe {
*sb.wrapping_offset(7 as isize) = (len as i32).wrapping_add(14) as u8;
}
desc = sb.wrapping_offset(8 as isize).wrapping_offset((len as i32) as isize);
} else {
if ((unsafe { desc.offset_from(sb as *mut u8) }) as i64) > 82i64 {
return;
}
}
unsafe {
*desc.wrapping_offset(0 as isize) = 9;
}
unsafe {
*desc.wrapping_offset(1 as isize) = 12;
}
// Copy registers into sense buffer.
unsafe {
*desc.wrapping_offset(2 as isize) = 0;
}
unsafe {
*desc.wrapping_offset(3 as isize) = (*tf).__slate_anon_8.error;
}
unsafe {
*desc.wrapping_offset(5 as isize) = (*tf).nsect;
}
unsafe {
*desc.wrapping_offset(7 as isize) = (*tf).lbal;
}
unsafe {
*desc.wrapping_offset(9 as isize) = (*tf).lbam;
}
unsafe {
*desc.wrapping_offset(11 as isize) = (*tf).lbah;
}
unsafe {
*desc.wrapping_offset(12 as isize) = (*tf).device;
}
unsafe {
*desc.wrapping_offset(13 as isize) = (*tf).__slate_anon_14.status;
}
// Fill in Extend bit, and the high order bytes
// if applicable.
if (unsafe { (*tf).flags }) & 1 != 0 {
let __v1: *mut u8 = desc.wrapping_offset(2 as isize);
unsafe {
*__v1 = ((*__v1 as i32) | 1) as u8;
}
unsafe {
*desc.wrapping_offset(4 as isize) = (*tf).hob_nsect;
}
unsafe {
*desc.wrapping_offset(6 as isize) = (*tf).hob_lbal;
}
unsafe {
*desc.wrapping_offset(8 as isize) = (*tf).hob_lbam;
}
unsafe {
*desc.wrapping_offset(10 as isize) = (*tf).hob_lbah;
}
}
} else {
// Fixed sense format
let __v4: *mut u8 = sb.wrapping_offset(0 as isize);
unsafe {
*__v4 = ((*__v4 as i32) | 128) as u8;
}
unsafe {
*sb.wrapping_offset(3 as isize) = (*tf).__slate_anon_8.error;
}
unsafe {
*sb.wrapping_offset(4 as isize) = (*tf).__slate_anon_14.status;
}
unsafe {
*sb.wrapping_offset(5 as isize) = (*tf).device;
}
unsafe {
*sb.wrapping_offset(6 as isize) = (*tf).nsect;
}
if (unsafe { (*tf).flags }) & 1 != 0 {
let __v7: *mut u8 = sb.wrapping_offset(8 as isize);
unsafe {
*__v7 = ((*__v7 as i32) | 128) as u8;
}
if (unsafe { (*tf).hob_nsect }) != 0 {
let __v10: *mut u8 = sb.wrapping_offset(8 as isize);
unsafe {
*__v10 = ((*__v10 as i32) | 64) as u8;
}
}
if (unsafe { (*tf).hob_lbal }) != 0 || (unsafe { (*tf).hob_lbam }) != 0 || (unsafe { (*tf).hob_lbah }) != 0 {
let __v13: *mut u8 = sb.wrapping_offset(8 as isize);
unsafe {
*__v13 = ((*__v13 as i32) | 32) as u8;
}
}
}
unsafe {
*sb.wrapping_offset(9 as isize) = (*tf).lbal;
}
unsafe {
*sb.wrapping_offset(10 as isize) = (*tf).lbam;
}
unsafe {
*sb.wrapping_offset(11 as isize) = (*tf).lbah;
}
}
}

fn ata_scsi_set_invalid_field(dev: *mut ata_device, cmd: *mut scsi_cmnd, field: u16, bit: u8) {
ata_scsi_set_sense(dev, cmd, 5u8, 36u8, 0u8);
// "Invalid field in CDB"
unsafe { scsi_set_sense_field_pointer((*cmd).sense_buffer, 96i32, field, bit, true) };
}

fn ata_scsi_set_invalid_parameter(dev: *mut ata_device, cmd: *mut scsi_cmnd, field: u16) {
// "Invalid field in parameter list"
ata_scsi_set_sense(dev, cmd, 5u8, 38u8, 0u8);
unsafe { scsi_set_sense_field_pointer((*cmd).sense_buffer, 96i32, field, 255u8, false) };
}

static mut ata_common_sdev_attrs: __SlateAlign16<[*mut attribute; 2]> = __SlateAlign16([unsafe { core::ptr::addr_of_mut!(dev_attr_unload_heads.attr) }, core::ptr::null_mut::<attribute>()]);

static ata_common_sdev_attr_group: __SlateConst<attribute_group> = __SlateConst({
let mut __t0: attribute_group = unsafe { core::mem::zeroed() };
__t0.__slate_anon_4 = __SlateRecord_sysfs_h_118 { attrs: unsafe { core::ptr::addr_of_mut!(ata_common_sdev_attrs.0) as *mut *mut attribute } };
__t0
});

#[unsafe(no_mangle)]
static mut ata_common_sdev_groups: __SlateAlign16<[*const attribute_group; 2]> = __SlateAlign16([unsafe { core::ptr::addr_of!(ata_common_sdev_attr_group.0) }, core::ptr::null::<attribute_group>()]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_common_sdev_groups_675: *mut () = (unsafe { core::ptr::addr_of_mut!(ata_common_sdev_groups.0) }) as *mut ();

/// ata_std_bios_param - generic bios head/sector/cylinder calculator used by sd.
/// @sdev: SCSI device for which BIOS geometry is to be determined
/// @unused: gendisk associated with @sdev
/// @capacity: capacity of SCSI device
/// @geom: location to which geometry will be output
///
/// Generic bios head/sector/cylinder calculator
/// used by sd. Most BIOSes nowadays expect a XXX/255/16  (CHS)
/// mapping. Some situations may arise where the disk is not
/// bootable if this is not used.
///
/// LOCKING:
/// Defined by the SCSI layer.  We don't really care.
///
/// RETURNS:
/// Zero.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_std_bios_param")]
extern "C" fn ata_std_bios_param(sdev: *mut scsi_device, unused: *mut gendisk, mut capacity: u64, geom: *mut i32) -> i32 {
unsafe {
*geom.wrapping_offset(0 as isize) = 255;
}
unsafe {
*geom.wrapping_offset(1 as isize) = 63;
}
sector_div!(capacity, 16065i32);
unsafe {
*geom.wrapping_offset(2 as isize) = capacity as i32;
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_std_bios_param_676: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, *mut gendisk, u64, *mut i32) -> i32>, *mut ()>(Some(ata_std_bios_param)) };

/// ata_scsi_unlock_native_capacity - unlock native capacity
/// @sdev: SCSI device to adjust device capacity for
///
/// This function is called if a partition on @sdev extends beyond
/// the end of the device.  It requests EH to unlock HPA.
///
/// LOCKING:
/// Defined by the SCSI layer.  Might sleep.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_unlock_native_capacity")]
extern "C" fn ata_scsi_unlock_native_capacity(sdev: *mut scsi_device) {
let ap: *mut ata_port = ata_shost_to_port(unsafe { (*sdev).host });
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
let dev: *mut ata_device = ata_scsi_find_dev(ap, sdev as *const scsi_device);
if !dev.is_null() && (unsafe { (*dev).n_sectors }) < unsafe { (*dev).n_native_sectors } {
unsafe {
(*dev).flags = (*dev).flags | 4194304;
}
let __v6: *mut ata_link = unsafe { (*dev).link };
unsafe {
(*__v6).eh_info.action = (*__v6).eh_info.action | 6;
}
unsafe { ata_port_schedule_eh(ap) };
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
unsafe { ata_port_wait_eh(ap) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_unlock_native_capacity_677: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device)>, *mut ()>(Some(ata_scsi_unlock_native_capacity)) };

/// ata_get_identity - Handler for HDIO_GET_IDENTITY ioctl
/// @ap: target port
/// @sdev: SCSI device to get identify data for
/// @arg: User buffer area for identify data
///
/// LOCKING:
/// Defined by the SCSI layer.  We don't really care.
///
/// RETURNS:
/// Zero on success, negative errno on error.
fn ata_get_identity(ap: *mut ata_port, sdev: *mut scsi_device, arg: *mut ()) -> i32 {
let dev: *mut ata_device = ata_scsi_find_dev(ap, sdev as *const scsi_device);
let dst: *mut u16 = arg as *mut u16;
let mut buf: __SlateAlign16<[u8; 40]> = __SlateAlign16([0u8; 40]);
if dev.is_null() {
return 42i32.wrapping_neg();
}
if copy_to_user(dst as *mut (), (unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const (), 256u64.wrapping_mul(core::mem::size_of::<u16>() as u64)) != 0 {
return 14i32.wrapping_neg();
}
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, buf.0.as_mut_ptr() as *mut u8, 27u32, 40u32) };
if copy_to_user(dst.wrapping_offset(27 as isize) as *mut (), buf.0.as_mut_ptr() as *const (), 40u64) != 0 {
return 14i32.wrapping_neg();
}
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, buf.0.as_mut_ptr() as *mut u8, 23u32, 8u32) };
if copy_to_user(dst.wrapping_offset(23 as isize) as *mut (), buf.0.as_mut_ptr() as *const (), 8u64) != 0 {
return 14i32.wrapping_neg();
}
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, buf.0.as_mut_ptr() as *mut u8, 10u32, 20u32) };
if copy_to_user(dst.wrapping_offset(10 as isize) as *mut (), buf.0.as_mut_ptr() as *const (), 20u64) != 0 {
return 14i32.wrapping_neg();
}
0
}

/// ata_cmd_ioctl - Handler for HDIO_DRIVE_CMD ioctl
/// @scsidev: Device to which we are issuing command
/// @arg: User provided data for issuing command
///
/// LOCKING:
/// Defined by the SCSI layer.  We don't really care.
///
/// RETURNS:
/// Zero on success, negative errno on error.
#[unsafe(no_mangle)]
extern "C" fn ata_cmd_ioctl(scsidev: *mut scsi_device, arg: *mut ()) -> i32 {
let mut __slate_compound_17: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_19: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __v0: bool = unsafe { core::mem::zeroed() };
let mut desc: *mut u8 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut cmd_result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(cmd_result) as *mut i32;
let mut exec_args: core::mem::MaybeUninit<scsi_exec_args> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut scsi_exec_args = core::ptr::addr_of_mut!(exec_args) as *mut scsi_exec_args;
let mut sshdr: core::mem::MaybeUninit<scsi_sense_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut scsi_sense_hdr = core::ptr::addr_of_mut!(sshdr) as *mut scsi_sense_hdr;
let mut argsize: i32 = unsafe { core::mem::zeroed() };
let mut argbuf: *mut u8 = unsafe { core::mem::zeroed() };
let mut args: core::mem::MaybeUninit<[u8; 4]> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut [u8; 4] = core::ptr::addr_of_mut!(args) as *mut [u8; 4];
let mut scsi_cmd: core::mem::MaybeUninit<__SlateAlign16<[u8; 16]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut [u8; 16] = core::ptr::addr_of_mut!(scsi_cmd) as *mut [u8; 16];
let mut sensebuf: core::mem::MaybeUninit<__SlateAlign16<[u8; 96]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut [u8; 96] = core::ptr::addr_of_mut!(sensebuf) as *mut [u8; 96];
let mut rc: i32 = unsafe { core::mem::zeroed() };
unsafe {
rc = 0;
let __slate_auto_bytes_2 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 96usize);
let __slate_auto_bytes_3 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_4 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
argbuf = core::ptr::null_mut::<u8>();
argsize = 0;
let __slate_auto_bytes_5 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_9, {
let mut __t0: scsi_exec_args = core::mem::zeroed();
__t0.sense = (*__slate_slot_15).as_mut_ptr() as *mut u8;
__t0.sense_len = core::mem::size_of::<[u8; 96]>() as u32;
__t0.sshdr = core::ptr::addr_of_mut!(*__slate_slot_10);
__t0
});
let __slate_auto_bytes_6 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
if arg.is_null() {
return 22i32.wrapping_neg();
} else {
if copy_from_user((*__slate_slot_13).as_mut_ptr() as *mut (), arg as *const (), core::mem::size_of::<[u8; 4]>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
'__join_0: {
memset((*__slate_slot_15).as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[u8; 96]>() as u64);
memset((*__slate_slot_14).as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[u8; 16]>() as u64);
if (*__slate_slot_13)[3] != 0 {
argsize = 512i32.wrapping_mul((*__slate_slot_13)[3] as i32);
let __slate_auto_bytes_1 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_4 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_5 = _kmalloc_noprof(argsize as u64, 3264u32, *{
__slate_compound_17 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_17)
});
alloc_tag_restore!();
} else {
*__slate_slot_5 = _kmalloc_noprof(argsize as u64, 3264u32, *{
__slate_compound_19 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_19)
});
}
argbuf = *__slate_slot_5 as *mut u8;
if argbuf.is_null() {
rc = 12i32.wrapping_neg();
break '__join_0;
} else {
(*__slate_slot_14)[1] = 8u8; // PIO Data-in
(*__slate_slot_14)[2] = 14;
// no off.line or cc, read from dev,
// block count in sector count field
}
} else {
(*__slate_slot_14)[1] = 6u8; // Non-data
(*__slate_slot_14)[2] = 32; // cc but no off.line or data xfer
}
(*__slate_slot_14)[0] = 133;
(*__slate_slot_14)[4] = (*__slate_slot_13)[2];
if ((*__slate_slot_13)[0] as i32) == 176 {
// hack -- ide driver does this too
(*__slate_slot_14)[6] = (*__slate_slot_13)[3];
(*__slate_slot_14)[8] = (*__slate_slot_13)[1];
(*__slate_slot_14)[10] = 79;
(*__slate_slot_14)[12] = 194;
} else {
(*__slate_slot_14)[6] = (*__slate_slot_13)[1];
}
(*__slate_slot_14)[14] = (*__slate_slot_13)[0];
// Good values for timeout and retries?  Values below
// from scsi_ioctl_send_command() for default case...
*__slate_slot_8 = scsi_execute_cmd(scsidev, (*__slate_slot_14).as_mut_ptr() as *const u8, 34u32, argbuf as *mut (), argsize as u32, 10000i32, 5i32, core::ptr::addr_of!(*__slate_slot_9));
if *__slate_slot_8 < 0 {
rc = *__slate_slot_8;
} else {
if scsi_sense_valid(core::ptr::addr_of_mut!(*__slate_slot_10) as *const scsi_sense_hdr) {
// sense data available
desc = ((*__slate_slot_15).as_mut_ptr() as *mut u8).wrapping_offset(8 as isize);
// If we set cc then ATA pass-through will cause a
// check condition even if no error. Filter that.
if scsi_status_is_check_condition(*__slate_slot_8) != 0 {
if ((*__slate_slot_10).sense_key as i32) == 1 && ((*__slate_slot_10).asc as i32) == 0 && ((*__slate_slot_10).ascq as i32) == 29 {
let __v1: i32 = *__slate_slot_8 & !2;
*__slate_slot_8 = __v1;
}
}
// Send userspace a few ATA registers (same as drivers/ide)
if ((*__slate_slot_15)[0] as i32) == 114 && (*desc.wrapping_offset(0 as isize) as i32) == 9 {
// code is "ATA Descriptor"
(*__slate_slot_13)[0] = *desc.wrapping_offset(13 as isize); // status
(*__slate_slot_13)[1] = *desc.wrapping_offset(3 as isize); // error
(*__slate_slot_13)[2] = *desc.wrapping_offset(5 as isize); // sector count (0:7)
if copy_to_user(arg, (*__slate_slot_13).as_mut_ptr() as *const (), core::mem::size_of::<[u8; 4]>() as u64) != 0 {
rc = 14i32.wrapping_neg();
}
}
// format is "descriptor"
}
if *__slate_slot_8 != 0 {
rc = 5i32.wrapping_neg();
} else {
if !argbuf.is_null() {
__v0 = copy_to_user((arg as *const u8).wrapping_offset(core::mem::size_of::<[u8; 4]>() as isize) as *mut (), argbuf as *const (), argsize as u64) != 0;
} else {
__v0 = false;
}
if __v0 {
rc = 14i32.wrapping_neg();
}
}
}
}
kfree(argbuf as *const ());
return rc;
}
}
}
}

/// ata_task_ioctl - Handler for HDIO_DRIVE_TASK ioctl
/// @scsidev: Device to which we are issuing command
/// @arg: User provided data for issuing command
///
/// LOCKING:
/// Defined by the SCSI layer.  We don't really care.
///
/// RETURNS:
/// Zero on success, negative errno on error.
#[unsafe(no_mangle)]
extern "C" fn ata_task_ioctl(scsidev: *mut scsi_device, arg: *mut ()) -> i32 {
let mut desc: *mut u8 = unsafe { core::mem::zeroed() };
let mut exec_args: core::mem::MaybeUninit<scsi_exec_args> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut scsi_exec_args = core::ptr::addr_of_mut!(exec_args) as *mut scsi_exec_args;
let mut cmd_result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut i32 = core::ptr::addr_of_mut!(cmd_result) as *mut i32;
let mut sshdr: core::mem::MaybeUninit<scsi_sense_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut scsi_sense_hdr = core::ptr::addr_of_mut!(sshdr) as *mut scsi_sense_hdr;
let mut args: core::mem::MaybeUninit<[u8; 7]> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut [u8; 7] = core::ptr::addr_of_mut!(args) as *mut [u8; 7];
let mut scsi_cmd: core::mem::MaybeUninit<__SlateAlign16<[u8; 16]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut [u8; 16] = core::ptr::addr_of_mut!(scsi_cmd) as *mut [u8; 16];
let mut sensebuf: core::mem::MaybeUninit<__SlateAlign16<[u8; 96]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut [u8; 96] = core::ptr::addr_of_mut!(sensebuf) as *mut [u8; 96];
let mut rc: i32 = unsafe { core::mem::zeroed() };
unsafe {
rc = 0;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 96usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 16usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 7usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_3, {
let mut __t0: scsi_exec_args = core::mem::zeroed();
__t0.sense = (*__slate_slot_8).as_mut_ptr() as *mut u8;
__t0.sense_len = core::mem::size_of::<[u8; 96]>() as u32;
__t0.sshdr = core::ptr::addr_of_mut!(*__slate_slot_5);
__t0
});
if arg.is_null() {
return 22i32.wrapping_neg();
} else {
if copy_from_user((*__slate_slot_6).as_mut_ptr() as *mut (), arg as *const (), core::mem::size_of::<[u8; 7]>() as u64) != 0 {
return 14i32.wrapping_neg();
} else {
memset((*__slate_slot_8).as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[u8; 96]>() as u64);
memset((*__slate_slot_7).as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[u8; 16]>() as u64);
(*__slate_slot_7)[0] = 133;
(*__slate_slot_7)[1] = 6u8; // Non-data
(*__slate_slot_7)[2] = 32; // cc but no off.line or data xfer
(*__slate_slot_7)[4] = (*__slate_slot_6)[1];
(*__slate_slot_7)[6] = (*__slate_slot_6)[2];
(*__slate_slot_7)[8] = (*__slate_slot_6)[3];
(*__slate_slot_7)[10] = (*__slate_slot_6)[4];
(*__slate_slot_7)[12] = (*__slate_slot_6)[5];
(*__slate_slot_7)[13] = (((*__slate_slot_6)[6] as i32) & 79) as u8;
(*__slate_slot_7)[14] = (*__slate_slot_6)[0];
// Good values for timeout and retries?  Values below
// from scsi_ioctl_send_command() for default case...
*__slate_slot_4 = scsi_execute_cmd(scsidev, (*__slate_slot_7).as_mut_ptr() as *const u8, 34u32, core::ptr::null_mut::<()>(), 0u32, 10000i32, 5i32, core::ptr::addr_of!(*__slate_slot_3));
if *__slate_slot_4 < 0 {
rc = *__slate_slot_4;
} else {
if scsi_sense_valid(core::ptr::addr_of_mut!(*__slate_slot_5) as *const scsi_sense_hdr) {
// sense data available
desc = ((*__slate_slot_8).as_mut_ptr() as *mut u8).wrapping_offset(8 as isize);
// If we set cc then ATA pass-through will cause a
// check condition even if no error. Filter that.
if *__slate_slot_4 & 2 != 0 {
if ((*__slate_slot_5).sense_key as i32) == 1 && ((*__slate_slot_5).asc as i32) == 0 && ((*__slate_slot_5).ascq as i32) == 29 {
let __v0: i32 = *__slate_slot_4 & !2;
*__slate_slot_4 = __v0;
}
}
// Send userspace ATA registers
if ((*__slate_slot_8)[0] as i32) == 114 && (*desc.wrapping_offset(0 as isize) as i32) == 9 {
// code is "ATA Descriptor"
(*__slate_slot_6)[0] = *desc.wrapping_offset(13 as isize); // status
(*__slate_slot_6)[1] = *desc.wrapping_offset(3 as isize); // error
(*__slate_slot_6)[2] = *desc.wrapping_offset(5 as isize); // sector count (0:7)
(*__slate_slot_6)[3] = *desc.wrapping_offset(7 as isize); // lbal
(*__slate_slot_6)[4] = *desc.wrapping_offset(9 as isize); // lbam
(*__slate_slot_6)[5] = *desc.wrapping_offset(11 as isize); // lbah
(*__slate_slot_6)[6] = *desc.wrapping_offset(12 as isize); // select
if copy_to_user(arg, (*__slate_slot_6).as_mut_ptr() as *const (), core::mem::size_of::<[u8; 7]>() as u64) != 0 {
rc = 14i32.wrapping_neg();
}
}
// format is "descriptor"
}
if *__slate_slot_4 != 0 {
rc = 5i32.wrapping_neg();
}
}
return rc;
}
}
}
}

fn ata_ioc32(ap: *mut ata_port) -> bool {
if (unsafe { (*ap).flags }) & 128 != 0 {
return true;
}
if (unsafe { (*ap).pflags }) & 1048576 != 0 {
return true;
}
false
}

/// This handles both native and compat commands, so anything added
/// here must have a compatible argument, or check in_compat_syscall()
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_sas_scsi_ioctl")]
extern "C" fn ata_sas_scsi_ioctl(ap: *mut ata_port, scsidev: *mut scsi_device, cmd: u32, arg: *mut ()) -> i32 {
let mut val: u64 = 0;
let mut rc: i32 = 22i32.wrapping_neg();
let mut flags: u64 = 0;
match cmd {
777 => {
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
val = ata_ioc32(ap) as u64;
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
if in_compat_syscall() {
let mut __v3: i64;
might_fault();
let mut __v4: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: u32 = 0;
let mut __x: u32 = val as u32;
let __ptr: *mut u32 = arg as *mut u32;
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t0: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 4 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t0, in("ax") __val_pu as u32, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
instrument_put_user__2!(__x, __ptr);
__v4 = __ret_pu as i64;
__v3 = __v4;
return __v3 as i32;
}
let mut __v5: i64;
might_fault();
let mut __v6: i64;
let mut __ret_pu: i32 = 0;
let mut __ptr_pu: *mut () = unsafe { core::mem::zeroed() };
let mut __val_pu: u64 = 0;
let mut __x: u64 = val;
let __ptr: *mut u64 = arg as *mut u64;
__chk_user_ptr!();
__ptr_pu = __ptr as *mut ();
__val_pu = __x;
unsafe {
let __t1: u64;
core::arch::asm!("mov %rbx, {1:r}\ncall __put_user_{0}\nmov {1:r}, %rbx", const 8 as u64, out(reg) _, inlateout("cx") __ptr_pu as u64 => __t1, in("ax") __val_pu as u64, options(att_syntax));
__ret_pu = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
instrument_put_user__3!(__x, __ptr);
__v6 = __ret_pu as i64;
__v5 = __v6;
return __v5 as i32;
}
804 => {
val = arg as u64;
rc = 0;
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
if (unsafe { (*ap).pflags }) & 2097152 != 0 {
if val != 0 {
unsafe {
(*ap).pflags = (*ap).pflags | 1048576;
}
} else {
let __v15: u32 = (unsafe { (*ap).pflags }) & 4293918719u32;
unsafe {
(*ap).pflags = __v15;
}
}
} else {
if val != ((ata_ioc32(ap) as i32) as u64) {
rc = 22i32.wrapping_neg();
}
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
return rc;
}
781 => {
return ata_get_identity(ap, scsidev, arg);
}
799 => {
let __v16: bool = !unsafe { capable(21i32) } || !unsafe { capable(17i32) };
if __v16 {
return 13i32.wrapping_neg();
}
return ata_cmd_ioctl(scsidev, arg);
}
798 => {
let __v17: bool = !unsafe { capable(21i32) } || !unsafe { capable(17i32) };
if __v17 {
return 13i32.wrapping_neg();
}
return ata_task_ioctl(scsidev, arg);
}
_ => {
rc = 25i32.wrapping_neg();
}
}
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_sas_scsi_ioctl_678: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, *mut scsi_device, u32, *mut ()) -> i32>, *mut ()>(Some(ata_sas_scsi_ioctl)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_ioctl")]
extern "C" fn ata_scsi_ioctl(scsidev: *mut scsi_device, cmd: u32, arg: *mut ()) -> i32 {
ata_sas_scsi_ioctl(ata_shost_to_port(unsafe { (*scsidev).host }), scsidev, cmd, arg)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_ioctl_679: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, u32, *mut ()) -> i32>, *mut ()>(Some(ata_scsi_ioctl)) };

/// ata_scsi_qc_new - acquire new ata_queued_cmd reference
/// @dev: ATA device to which the new command is attached
/// @cmd: SCSI command that originated this ATA command
///
/// Obtain a reference to an unused ata_queued_cmd structure,
/// which is the basic libata structure representing a single
/// ATA command sent to the hardware.
///
/// If a command was available, fill in the SCSI-specific
/// portions of the structure with information on the
/// current command.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// Command allocated, or %NULL if none available.
fn ata_scsi_qc_new(dev: *mut ata_device, cmd: *mut scsi_cmnd) -> *mut ata_queued_cmd {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut tag: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(tag) as *mut i32;
let mut qc: core::mem::MaybeUninit<*mut ata_queued_cmd> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut ata_queued_cmd = core::ptr::addr_of_mut!(qc) as *mut *mut ata_queued_cmd;
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
ap = (*(*dev).link).ap;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if unlikely!(ata_port_is_frozen(ap as *const ata_port)) == 0 {
if (*ap).flags & 16777216 != 0 {
__ret_warn_on = ((*cmd).budget_token >= 32) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("680: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 680b - ., 3; .popsection\n# {0}", const 680 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_scsi_qc_new_10, sym __slate_asm_string_ata_scsi_qc_new_11, const 761 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("681: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 681b - ., 4; .popsection\n# {0}", const 681 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
// SAS hosts may queue > ATA_MAX_QUEUE commands so use
// unique per-device budget token as a tag.
if __ret_warn_on != 0 {
break '__join_0;
} else {
*__slate_slot_7 = (*cmd).budget_token;
}
} else {
*__slate_slot_7 = (*scsi_cmd_to_rq(cmd)).tag;
}
*__slate_slot_8 = __ata_qc_from_tag(ap, *__slate_slot_7 as u32);
let __v3: u32 = *__slate_slot_7 as u32;
(*(*__slate_slot_8)).hw_tag = __v3;
(*(*__slate_slot_8)).tag = __v3;
(*(*__slate_slot_8)).ap = ap;
(*(*__slate_slot_8)).dev = dev;
ata_qc_reinit(*__slate_slot_8);
(*(*__slate_slot_8)).scsicmd = cmd;
(*(*__slate_slot_8)).scsidone = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut scsi_cmnd)>>(scsi_done as *const ()) };
(*(*__slate_slot_8)).sg = scsi_sglist(cmd);
(*(*__slate_slot_8)).n_elem = scsi_sg_count(cmd);
if (*scsi_cmd_to_rq(cmd)).rq_flags & 128u32 != 0 {
let __v2: *mut ata_queued_cmd = *__slate_slot_8;
(*__v2).flags = (*__v2).flags | 64;
}
return *__slate_slot_8;
}
}
set_host_byte(cmd, 0u8);
set_status_byte(cmd, 40u8);
scsi_done(cmd);
return core::ptr::null_mut::<ata_queued_cmd>();
}
}

fn ata_qc_set_pc_nbytes(qc: *mut ata_queued_cmd) {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
unsafe {
(*qc).extrabytes = (*scmd).extra_len;
}
unsafe {
(*qc).nbytes = scsi_bufflen(scmd).wrapping_add((*qc).extrabytes);
}
}

/// ata_to_sense_error - convert ATA error to SCSI error
/// @drv_stat: value contained in ATA status register
/// @drv_err: value contained in ATA error register
/// @sk: the sense key we'll fill out
/// @asc: the additional sense code we'll fill out
/// @ascq: the additional sense code qualifier we'll fill out
///
/// Converts an ATA error into a SCSI error.  Fill out pointers to
/// SK, ASC, and ASCQ bytes for later use in fixed or descriptor
/// format sense blocks.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_to_sense_error(drv_stat: u8, mut drv_err: u8, sk: &mut u8, asc: &mut u8, ascq: &mut u8) {
let mut i: i32 = 0;
// Based on the 3ware driver translation table
// BBD|ECC|ID|MAR
// Device busy                  Aborted command
//
// BBD|ECC|ID
// Device busy                  Aborted command
//
// ECC|MC|MARK
// Device fault                 Hardware error
//
// ICRC|ABRT
//
// NB: ICRC & !ABRT is BBD
// Data CRC error               SCSI parity error
//
// MC|ID|ABRT|TRK0|MARK
// Unit offline                 Not ready
//
// MCR|MARK
// Unrecovered disk error       Not ready
//
//  Bad address mark
// Address mark not found for data field
//
// TRK0 - Track 0 not found
// Hardware error
//
// Abort: 0x04 is not translated here, see below
//
// Media change request
// FIXME: faking offline
//
// SRV/IDNF - ID not found
// Logical address out of range
//
// MC - Media Changed
// Not ready to ready change, medium may have changed
//
// ECC - Uncorrectable ECC error
// Unrecovered read error
//
// BBD - block marked bad
// Block marked bad	Medium error, unrecovered read error
// END mark
// Busy: must be first because BUSY means no other bits valid
// Device fault: INTERNAL TARGET FAILURE
// Corrected data error
// END mark
// Is this an error we can process/parse
if (drv_stat as i32) & 128 != 0 {
drv_err = 0; // Ignore the err bits, they're invalid
}
if drv_err != 0 {
// Look for drv_err
i = 0;
while ((unsafe { sense_table.0[i as usize][0] }) as i32) != 255 {
// Look for best matches first
if ((unsafe { sense_table.0[i as usize][0] }) as i32) & (drv_err as i32) == ((unsafe { sense_table.0[i as usize][0] }) as i32) {
unsafe {
*(sk as *mut u8) = sense_table.0[i as usize][1];
}
unsafe {
*(asc as *mut u8) = sense_table.0[i as usize][2];
}
unsafe {
*(ascq as *mut u8) = sense_table.0[i as usize][3];
}
return;
}
i = i.wrapping_add(1);
}
}
// Fall back to interpreting status bits.  Note that if the drv_err
// has only the ABRT bit set, we decode drv_stat.  ABRT by itself
// is not descriptive enough.
i = 0;
while ((unsafe { stat_table.0[i as usize][0] }) as i32) != 255 {
if ((unsafe { stat_table.0[i as usize][0] }) as i32) & (drv_stat as i32) != 0 {
unsafe {
*(sk as *mut u8) = stat_table.0[i as usize][1];
}
unsafe {
*(asc as *mut u8) = stat_table.0[i as usize][2];
}
unsafe {
*(ascq as *mut u8) = stat_table.0[i as usize][3];
}
return;
}
i = i.wrapping_add(1);
}
// We need a sensible error return here, which is tricky, and one
// that won't cause people to do things like return a disk wrongly.
unsafe {
*(sk as *mut u8) = 11;
}
unsafe {
*(asc as *mut u8) = 0;
}
unsafe {
*(ascq as *mut u8) = 0;
}
}

static sense_table: __SlateAlign16<[[u8; 4]; 14]> = __SlateAlign16([[209u8, 11u8, 0u8, 0u8], [208u8, 11u8, 0u8, 0u8], [97u8, 4u8, 0u8, 0u8], [132u8, 11u8, 71u8, 0u8], [55u8, 2u8, 4u8, 0u8], [9u8, 2u8, 4u8, 0u8], [1u8, 3u8, 19u8, 0u8], [2u8, 4u8, 0u8, 0u8], [8u8, 2u8, 4u8, 0u8], [16u8, 5u8, 33u8, 0u8], [32u8, 6u8, 40u8, 0u8], [64u8, 3u8, 17u8, 4u8], [128u8, 3u8, 17u8, 4u8], [255u8, 255u8, 255u8, 255u8]]);

static stat_table: __SlateAlign16<[[u8; 4]; 4]> = __SlateAlign16([[128u8, 11u8, 0u8, 0u8], [32u8, 4u8, 68u8, 0u8], [4u8, 1u8, 0u8, 0u8], [255u8, 255u8, 255u8, 255u8]]);

/// ata_gen_passthru_sense - Generate check condition sense block.
/// @qc: Command that completed.
///
/// This function is specific to the ATA pass through commands.
/// Regardless of whether the command errored or not, return a sense
/// block. If there was no error, we get the request from an ATA
/// passthrough command, so we use the following sense data:
/// sk = RECOVERED ERROR
/// asc,ascq = ATA PASS-THROUGH INFORMATION AVAILABLE
///
///
/// LOCKING:
/// None.
fn ata_gen_passthru_sense(qc: *mut ata_queued_cmd) {
let dev: *mut ata_device = unsafe { (*qc).dev };
let cmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).result_tf) };
let mut sense_key: u8 = 0;
let mut asc: u8 = 0;
let mut ascq: u8 = 0;
if (unsafe { (*qc).flags }) & 4 == 0 {
let mut __v0: i32;
if false {
unsafe { _printk(c"\x017ata%u.%02u: missing result TF: can't generate ATA PT sense data\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
}
__v0 = 0;
if (unsafe { (*qc).err_mask }) != 0 {
ata_scsi_set_sense(dev, cmd, 11u8, 0u8, 0u8);
}
return;
}
// Use ata_to_sense_error() to map status register bits
// onto sense key, asc & ascq.
if (unsafe { (*qc).err_mask }) != 0 || ((unsafe { (*tf).__slate_anon_14.status }) as i32) & (128 | 32 | 1 | 8) != 0 {
ata_to_sense_error(unsafe { (*tf).__slate_anon_14.status }, unsafe { (*tf).__slate_anon_8.error }, unsafe { &mut (*core::ptr::addr_of_mut!(sense_key)) }, unsafe { &mut (*core::ptr::addr_of_mut!(asc)) }, unsafe { &mut (*core::ptr::addr_of_mut!(ascq)) });
ata_scsi_set_sense(unsafe { (*qc).dev }, cmd, sense_key, asc, ascq);
} else {
// ATA PASS-THROUGH INFORMATION AVAILABLE
//
// Note: we are supposed to call ata_scsi_set_sense(), which
// respects the D_SENSE bit, instead of unconditionally
// generating the sense data in descriptor format. However,
// because hdparm, hddtemp, and udisks incorrectly assume sense
// data in descriptor format, without even looking at the
// RESPONSE CODE field in the returned sense data (to see which
// format the returned sense data is in), we are stuck with
// being bug compatible with older kernels.
unsafe { scsi_build_sense(cmd, 1i32, 1u8, 0u8, 29u8) };
}
}

/// ata_gen_ata_sense - generate a SCSI fixed sense block
/// @qc: Command that we are erroring out
///
/// Generate sense block for a failed ATA command @qc.
///
/// LOCKING:
/// None.
fn ata_gen_ata_sense(qc: *mut ata_queued_cmd) {
let mut ascq: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut u8 = core::ptr::addr_of_mut!(ascq) as *mut u8;
let mut asc: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut u8 = core::ptr::addr_of_mut!(asc) as *mut u8;
let mut sense_key: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut u8 = core::ptr::addr_of_mut!(sense_key) as *mut u8;
let mut cmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
unsafe {
dev = (*qc).dev;
cmd = (*qc).scsicmd;
let tf: *mut ata_taskfile = core::ptr::addr_of_mut!((*qc).result_tf);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
if ata_dev_disabled(dev as *const ata_device) != 0 {
// Device disabled after error recovery
//
// LOGICAL UNIT NOT READY, HARD RESET REQUIRED
ata_scsi_set_sense(dev, cmd, 2u8, 4u8, 33u8);
return;
} else {
if ata_id_is_locked!((*dev).__slate_anon_26.id) {
// Security locked
//
// LOGICAL UNIT ACCESS NOT AUTHORIZED
ata_scsi_set_sense(dev, cmd, 7u8, 116u8, 113u8);
return;
} else {
if (*qc).flags & 4 != 0 {
// Use ata_to_sense_error() to map status register bits
// onto sense key, asc & ascq.
if (*qc).err_mask != 0 || ((*tf).__slate_anon_14.status as i32) & (128 | 32 | 1 | 8) != 0 {
ata_to_sense_error((*tf).__slate_anon_14.status, (*tf).__slate_anon_8.error, &mut (*core::ptr::addr_of_mut!(*__slate_slot_4)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_3)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_2)));
ata_scsi_set_sense(dev, cmd, *__slate_slot_4, *__slate_slot_3, *__slate_slot_2);
return;
} else {
while false {
}
// Could not decode error
_printk(c"\x014ata%u.%02u: Could not decode error 0x%x, status 0x%x (err_mask=0x%x)\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno), (*tf).__slate_anon_8.error as i32, (*tf).__slate_anon_14.status as i32, (*qc).err_mask);
}
}
ata_scsi_set_sense(dev, cmd, 11u8, 0u8, 0u8);
}
}
}
}

#[unsafe(no_mangle)]
extern "C" fn ata_scsi_sdev_config(sdev: *mut scsi_device) {
unsafe {
(*sdev).__slate_bits_0.__set_use_10_for_rw(1u32);
}
unsafe {
(*sdev).__slate_bits_0.__set_use_10_for_ms(1u32);
}
unsafe {
(*sdev).__slate_bits_0.__set_no_write_same(1u32);
}
// Schedule policy is determined by ->qc_defer() callback and
// it needs to see every deferred qc.  Set dev_blocked to 1 to
// prevent SCSI midlayer from automatically deferring
// requests.
unsafe {
(*sdev).max_device_blocked = 1;
}
}

/// ata_scsi_dma_need_drain - Check whether data transfer may overflow
/// @rq: request to be checked
///
/// ATAPI commands which transfer variable length data to host
/// might overflow due to application error or hardware bug.  This
/// function checks whether overflow should be drained and ignored
/// for @request.
///
/// LOCKING:
/// None.
///
/// RETURNS:
/// 1 if ; otherwise, 0.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_dma_need_drain")]
extern "C" fn ata_scsi_dma_need_drain(rq: *mut request) -> bool {
let scmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
(unsafe { atapi_cmd_type(*((*scmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize)) }) == 4
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_dma_need_drain_682: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request) -> bool>, *mut ()>(Some(ata_scsi_dma_need_drain)) };

// ata_scsi_sdev_init - Early setup of SCSI device
// @sdev: SCSI device to examine
//
// This is called from scsi_alloc_sdev() when the scsi device
// associated with an ATA device is scanned on a port.
//
// LOCKING:
// Defined by SCSI layer.  We don't really care.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_sdev_init")]
extern "C" fn ata_scsi_sdev_init(sdev: *mut scsi_device) -> i32 {
let ap: *mut ata_port = ata_shost_to_port(unsafe { (*sdev).host });
ata_scsi_sdev_config(sdev);
// Create a link from the ata_port device to the scsi device to ensure
// that PM does suspend/resume in the correct order: the scsi device is
// consumer (child) and the ata port the supplier (parent).
let link: *mut device_link = unsafe { device_link_add(core::ptr::addr_of_mut!((*sdev).sdev_gendev), core::ptr::addr_of_mut!((*ap).tdev), (BIT!(0i32) | BIT!(2i32) | BIT!(3i32)) as u32) };
if link.is_null() {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013ata%u: Failed to create link to scsi device %s\n".as_ptr() as *const u8, (*ap).print_id, dev_name(core::ptr::addr_of_mut!((*sdev).sdev_gendev) as *const device)) };
return 19i32.wrapping_neg();
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_sdev_init_692: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device) -> i32>, *mut ()>(Some(ata_scsi_sdev_init)) };

// ata_scsi_sdev_configure - Set SCSI device attributes
// @sdev: SCSI device to examine
// @lim: queue limits
//
// This is called before we actually start reading
// and writing to the device, to configure certain
// SCSI mid-layer behaviors.
//
// LOCKING:
// Defined by SCSI layer.  We don't really care.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_sdev_configure")]
extern "C" fn ata_scsi_sdev_configure(sdev: *mut scsi_device, lim: *mut queue_limits) -> i32 {
let ap: *mut ata_port = ata_shost_to_port(unsafe { (*sdev).host });
let dev: *mut ata_device = __ata_scsi_find_dev(ap, sdev as *const scsi_device);
if !dev.is_null() {
return ata_scsi_dev_config(sdev, lim, dev);
}
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_sdev_configure_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, *mut queue_limits) -> i32>, *mut ()>(Some(ata_scsi_sdev_configure)) };

/// ata_scsi_sdev_destroy - SCSI device is about to be destroyed
/// @sdev: SCSI device to be destroyed
///
/// @sdev is about to be destroyed for hot/warm unplugging.  If
/// this unplugging was initiated by libata as indicated by NULL
/// dev->sdev, this function doesn't have to do anything.
/// Otherwise, SCSI layer initiated warm-unplug is in progress.
/// Clear dev->sdev, schedule the device for ATA detach and invoke
/// EH.
///
/// LOCKING:
/// Defined by SCSI layer.  We don't really care.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_sdev_destroy")]
extern "C" fn ata_scsi_sdev_destroy(sdev: *mut scsi_device) {
let ap: *mut ata_port = ata_shost_to_port(unsafe { (*sdev).host });
let mut flags: u64 = 0;
unsafe { device_link_remove(core::ptr::addr_of_mut!((*sdev).sdev_gendev) as *mut (), core::ptr::addr_of_mut!((*ap).tdev)) };
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
let dev: *mut ata_device = __ata_scsi_find_dev(ap, sdev as *const scsi_device);
if !dev.is_null() && !unsafe { (*dev).sdev }.is_null() {
// SCSI device already in CANCEL state, no need to offline it
unsafe {
(*dev).sdev = core::ptr::null_mut::<scsi_device>();
}
unsafe {
(*dev).flags = (*dev).flags | 67108864;
}
unsafe { ata_port_schedule_eh(ap) };
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
unsafe { kfree((*sdev).dma_drain_buf as *const ()) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_sdev_destroy_694: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device)>, *mut ()>(Some(ata_scsi_sdev_destroy)) };

/// ata_scsi_start_stop_xlat - Translate SCSI START STOP UNIT command
/// @qc: Storage for translated ATA taskfile
///
/// Sets up an ATA taskfile to issue STANDBY (to stop) or READ VERIFY
/// (to start). Perhaps these commands should be preceded by
/// CHECK POWER MODE to see what power mode the device is already in.
/// [See SAT revision 5 at www.t10.org]
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// Zero on success, non-zero on error.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_start_stop_xlat")]
extern "C" fn ata_scsi_start_stop_xlat(qc: *mut ata_queued_cmd) -> u32 {
let mut bp: u8 = unsafe { core::mem::zeroed() };
let mut fp: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut u16 = core::ptr::addr_of_mut!(fp) as *mut u16;
let mut cdb: *const u8 = unsafe { core::mem::zeroed() };
let mut scmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
unsafe {
scmd = (*qc).scsicmd;
cdb = ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const u8;
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
bp = 255;
if ((*scmd).cmd_len as i32) < 5 {
*__slate_slot_1 = 4;
} else {
// LOEJ bit set not supported
if (*cdb.wrapping_offset(4 as isize) as i32) & 2 != 0 {
*__slate_slot_1 = 4;
bp = 1;
} else {
// Power conditions not supported
if (*cdb.wrapping_offset(4 as isize) as i32) >> 4i32 & 15 != 0 {
*__slate_slot_1 = 4;
bp = 3;
} else {
// Ignore IMMED bit (cdb[1] & 0x1), violates sat-r05
if !ata_dev_power_init_tf((*qc).dev, core::ptr::addr_of_mut!((*qc).tf), (*cdb.wrapping_offset(4 as isize) as i32) & 1 != 0) {
ata_scsi_set_sense((*qc).dev, scmd, 11u8, 0u8, 0u8);
return 1;
} else {
// Standby and Idle condition timers could be implemented but that
// would require libata to implement the Power condition mode page
// and allow the user to change it. Changing mode pages requires
// MODE SELECT to be implemented.
return 0;
}
}
}
}
ata_scsi_set_invalid_field((*qc).dev, scmd, *__slate_slot_1, bp);
return 1;
}
}

/// ata_scsi_flush_xlat - Translate SCSI SYNCHRONIZE CACHE command
/// @qc: Storage for translated ATA taskfile
///
/// Sets up an ATA taskfile to issue FLUSH CACHE or
/// FLUSH CACHE EXT.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// Zero on success, non-zero on error.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_flush_xlat")]
extern "C" fn ata_scsi_flush_xlat(qc: *mut ata_queued_cmd) -> u32 {
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).tf) };
unsafe {
(*tf).flags = (*tf).flags | 4;
}
unsafe {
(*tf).protocol = 0;
}
if (unsafe { (*(*qc).dev).flags }) & 16 != 0 {
unsafe {
(*tf).__slate_anon_14.command = 234;
}
} else {
unsafe {
(*tf).__slate_anon_14.command = 231;
}
}
// flush is critical for IO integrity, consider it an IO command
unsafe {
(*qc).flags = (*qc).flags | 8;
}
0
}

/// scsi_6_lba_len - Get LBA and transfer length
/// @cdb: SCSI command to translate
///
/// Calculate LBA and transfer length for 6-byte commands.
///
/// RETURNS:
/// @plba: the LBA
/// @plen: the transfer length
fn scsi_6_lba_len(cdb: *const u8, plba: &mut u64, plen: &mut u32) {
unsafe {
*(plba as *mut u64) = (get_unaligned_be24(cdb.wrapping_offset(1 as isize) as *const ()) & 2097151) as u64;
}
unsafe {
*(plen as *mut u32) = *cdb.wrapping_offset(4 as isize) as u32;
}
}

/// scsi_10_lba_len - Get LBA and transfer length
/// @cdb: SCSI command to translate
///
/// Calculate LBA and transfer length for 10-byte commands.
///
/// RETURNS:
/// @plba: the LBA
/// @plen: the transfer length
fn scsi_10_lba_len(cdb: *const u8, plba: &mut u64, plen: &mut u32) {
unsafe {
*(plba as *mut u64) = get_unaligned_be32(cdb.wrapping_offset(2 as isize) as *const ()) as u64;
}
unsafe {
*(plen as *mut u32) = get_unaligned_be16(cdb.wrapping_offset(7 as isize) as *const ()) as u32;
}
}

/// scsi_16_lba_len - Get LBA and transfer length
/// @cdb: SCSI command to translate
///
/// Calculate LBA and transfer length for 16-byte commands.
///
/// RETURNS:
/// @plba: the LBA
/// @plen: the transfer length
fn scsi_16_lba_len(cdb: *const u8, plba: &mut u64, plen: &mut u32) {
unsafe {
*(plba as *mut u64) = get_unaligned_be64(cdb.wrapping_offset(2 as isize) as *const ());
}
unsafe {
*(plen as *mut u32) = get_unaligned_be32(cdb.wrapping_offset(10 as isize) as *const ());
}
}

/// scsi_dld - Get duration limit descriptor index
/// @cdb: SCSI command to translate
///
/// Returns the dld bits indicating the index of a command duration limit
/// descriptor.
fn scsi_dld(cdb: *const u8) -> i32 {
(((unsafe { *cdb.wrapping_offset(1 as isize) }) as i32) & 1) << 2i32 | ((unsafe { *cdb.wrapping_offset(14 as isize) }) as i32) >> 6i32 & 3
}

/// ata_scsi_verify_xlat - Translate SCSI VERIFY command into an ATA one
/// @qc: Storage for translated ATA taskfile
///
/// Converts SCSI VERIFY command to an ATA READ VERIFY command.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// Zero on success, non-zero on error.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_verify_xlat")]
extern "C" fn ata_scsi_verify_xlat(qc: *mut ata_queued_cmd) -> u32 {
let mut track: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(track) as *mut u32;
let mut cyl: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u32 = core::ptr::addr_of_mut!(cyl) as *mut u32;
let mut head: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u32 = core::ptr::addr_of_mut!(head) as *mut u32;
// CHS
let mut sect: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut u32 = core::ptr::addr_of_mut!(sect) as *mut u32;
let mut fp: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut u16 = core::ptr::addr_of_mut!(fp) as *mut u16;
let mut n_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_23: *mut u32 = core::ptr::addr_of_mut!(n_block) as *mut u32;
let mut block: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u64 = core::ptr::addr_of_mut!(block) as *mut u64;
let mut cdb: *const u8 = unsafe { core::mem::zeroed() };
let mut dev_sectors: u64 = unsafe { core::mem::zeroed() };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
let mut tf: *mut ata_taskfile = unsafe { core::mem::zeroed() };
let mut scmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
unsafe {
'__join_2: {
{
scmd = (*qc).scsicmd;
tf = core::ptr::addr_of_mut!((*qc).tf);
dev = (*qc).dev;
dev_sectors = (*(*qc).dev).n_sectors;
cdb = ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const u8;
let __slate_auto_bytes_3 = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_23 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
(*tf).flags = (*tf).flags | 6u64;
(*tf).protocol = 0;
let __t0: i32 = *cdb.wrapping_offset(0 as isize) as i32;
if __t0 == 47 {
if ((*scmd).cmd_len as i32) < 10 {
*__slate_slot_22 = 9;
break '__join_2;
} else {
scsi_10_lba_len(cdb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_24)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_23)));
}
} else {
if __t0 == 143 {
if ((*scmd).cmd_len as i32) < 16 {
*__slate_slot_22 = 15;
break '__join_2;
} else {
scsi_16_lba_len(cdb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_24)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_23)));
}
} else {
*__slate_slot_22 = 0;
break '__join_2;
}
}
}
if *__slate_slot_23 == 0 {
(*scmd).result = 0;
return 1;
} else {
'__join_1: {
if *__slate_slot_24 < dev_sectors {
if (*__slate_slot_24).wrapping_add(*__slate_slot_23 as u64) <= dev_sectors {
if (*dev).flags & 1 != 0 {
(*tf).flags = (*tf).flags | 16;
if lba_28_ok(*__slate_slot_24, *__slate_slot_23) {
// use LBA28
(*tf).__slate_anon_14.command = 64;
(*tf).device = ((((*tf).device as i32) as u64) | *__slate_slot_24 >> 24i32 & 15) as u8;
} else {
if lba_48_ok(*__slate_slot_24, *__slate_slot_23) {
if (*dev).flags & 2 == 0 {
break '__join_1;
} else {
// use LBA48
(*tf).flags = (*tf).flags | 1;
(*tf).__slate_anon_14.command = 66;
(*tf).hob_nsect = (*__slate_slot_23 >> 8i32 & 255) as u8;
(*tf).hob_lbah = (*__slate_slot_24 >> 40i32 & 255) as u8;
(*tf).hob_lbam = (*__slate_slot_24 >> 32i32 & 255) as u8;
(*tf).hob_lbal = (*__slate_slot_24 >> 24i32 & 255) as u8;
}
} else {
break '__join_1;
}
}
// request too large even for LBA48
(*tf).nsect = (*__slate_slot_23 & 255) as u8;
(*tf).lbah = (*__slate_slot_24 >> 16i32 & 255) as u8;
(*tf).lbam = (*__slate_slot_24 >> 8i32 & 255) as u8;
(*tf).lbal = (*__slate_slot_24 & 255) as u8;
(*tf).device = (((*tf).device as i32) | 64) as u8;
} else {
let __slate_auto_bytes_5 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if !lba_28_ok(*__slate_slot_24, *__slate_slot_23) {
break '__join_1;
} else {
// Convert LBA to CHS
*__slate_slot_15 = (*__slate_slot_24 as u32) / (((*dev).sectors as i32) as u32);
*__slate_slot_16 = *__slate_slot_15 / (((*dev).heads as i32) as u32);
*__slate_slot_17 = *__slate_slot_15 % (((*dev).heads as i32) as u32);
*__slate_slot_18 = ((*__slate_slot_24 as u32) % (((*dev).sectors as i32) as u32)).wrapping_add(1);
// Check whether the converted CHS can fit.
// Cylinder: 0-65535
// Head: 0-15
// Sector: 1-255
if *__slate_slot_16 >> 16i32 != 0 || *__slate_slot_17 >> 4i32 != 0 || *__slate_slot_18 >> 8i32 != 0 || *__slate_slot_18 == 0 {
break '__join_1;
} else {
(*tf).__slate_anon_14.command = 64;
(*tf).nsect = (*__slate_slot_23 & 255) as u8;
// Sector count 0 means 256 sectors
(*tf).lbal = *__slate_slot_18 as u8;
(*tf).lbam = *__slate_slot_16 as u8;
(*tf).lbah = (*__slate_slot_16 >> 8i32) as u8;
(*tf).device = ((((*tf).device as i32) as u32) | *__slate_slot_17) as u8;
}
}
}
return 0;
}
}
}
ata_scsi_set_sense((*qc).dev, scmd, 5u8, 33u8, 0u8);
// "Logical Block Address out of range"
return 1;
}
}
ata_scsi_set_invalid_field((*qc).dev, scmd, *__slate_slot_22, 255u8);
return 1;
}
}

fn ata_check_nblocks(scmd: *mut scsi_cmnd, n_blocks: u32) -> bool {
let rq: *mut request = scsi_cmd_to_rq(scmd);
let mut req_blocks: u32 = 0;
if !blk_rq_is_passthrough(rq) {
return true;
}
req_blocks = blk_rq_bytes(rq as *const request) / unsafe { (*(*scmd).device).sector_size };
if n_blocks > req_blocks {
return false;
}
true
}

/// ata_scsi_rw_xlat - Translate SCSI r/w command into an ATA one
/// @qc: Storage for translated ATA taskfile
///
/// Converts any of six SCSI read/write commands into the
/// ATA counterpart, including starting sector (LBA),
/// sector count, and taking into account the device's LBA48
/// support.
///
/// Commands %READ_6, %READ_10, %READ_16, %WRITE_6, %WRITE_10, and
/// %WRITE_16 are currently supported.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// Zero on success, non-zero on error.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_rw_xlat")]
extern "C" fn ata_scsi_rw_xlat(qc: *mut ata_queued_cmd) -> u32 {
let mut fp: u16 = unsafe { core::mem::zeroed() };
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut n_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(n_block) as *mut u32;
let mut block: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u64 = core::ptr::addr_of_mut!(block) as *mut u64;
let mut dld: i32 = unsafe { core::mem::zeroed() };
let mut tf_flags: u32 = unsafe { core::mem::zeroed() };
let mut class: i32 = unsafe { core::mem::zeroed() };
let mut cdb: *const u8 = unsafe { core::mem::zeroed() };
let mut scmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
unsafe {
'__join_24: {
scmd = (*qc).scsicmd;
cdb = ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const u8;
let rq: *mut request = scsi_cmd_to_rq(scmd);
class = IOPRIO_PRIO_CLASS!(req_get_ioprio(rq));
tf_flags = 0;
dld = 0;
let __slate_auto_bytes = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
fp = 0;
let __t1: i32 = *cdb.wrapping_offset(0 as isize) as i32;
if __t1 != 10 {
if __t1 != 42 {
if __t1 != 138 {
break '__join_24;
}
}
}
tf_flags |= 8;
}
'__join_2: {
'__join_7: {
'__join_23: {
'__join_18: {
// Calculate the SCSI LBA, transfer length and FUA.
let __t0: i32 = *cdb.wrapping_offset(0 as isize) as i32;
if __t0 == 40 {
break '__join_23;
} else {
if __t0 == 42 {
break '__join_23;
} else {
if __t0 == 8 {
break '__join_18;
} else {
if __t0 == 10 {
break '__join_18;
} else {
if __t0 != 136 {
if __t0 != 138 {
fp = 0;
break '__join_2;
}
}
}
}
}
}
if unlikely!(((*scmd).cmd_len as i32) < 16) != 0 {
fp = 15;
break '__join_2;
} else {
scsi_16_lba_len(cdb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_12)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_11)));
dld = scsi_dld(cdb);
if (*cdb.wrapping_offset(1 as isize) as i32) & 8i32 != 0 {
tf_flags |= 32;
}
if !ata_check_nblocks(scmd, *__slate_slot_11) {
break '__join_2;
} else {
break '__join_7;
}
}
}
if unlikely!(((*scmd).cmd_len as i32) < 6) != 0 {
fp = 5;
break '__join_2;
} else {
scsi_6_lba_len(cdb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_12)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_11)));
// for 6-byte r/w commands, transfer length 0
// means 256 blocks of data, not 0 block.
if *__slate_slot_11 == 0 {
*__slate_slot_11 = 256;
}
if !ata_check_nblocks(scmd, *__slate_slot_11) {
break '__join_2;
} else {
break '__join_7;
}
}
}
if unlikely!(((*scmd).cmd_len as i32) < 10) != 0 {
fp = 9;
break '__join_2;
} else {
scsi_10_lba_len(cdb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_12)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_11)));
if (*cdb.wrapping_offset(1 as isize) as i32) & 8i32 != 0 {
tf_flags |= 32;
}
if !ata_check_nblocks(scmd, *__slate_slot_11) {
break '__join_2;
}
}
}
// Check and compose ATA command
if *__slate_slot_11 == 0 {
(*scmd).result = 0;
return 1;
} else {
// For 10-byte and 16-byte SCSI R/W commands, transfer
// length 0 means transfer 0 block of data.
// However, for ATA R/W commands, sector count 0 means
// 256 or 65536 sectors, not 0 sectors as in SCSI.
//
// WARNING: one or two older ATA drives treat 0 as 0...
(*qc).flags = (*qc).flags | 8;
(*qc).nbytes = (*__slate_slot_11).wrapping_mul((*(*scmd).device).sector_size);
*__slate_slot_10 = ata_build_rw_tf(qc, *__slate_slot_12, *__slate_slot_11, tf_flags, dld, class);
if likely!(*__slate_slot_10 == 0) != 0 {
return 0;
} else {
if *__slate_slot_10 == 34i32.wrapping_neg() {
ata_scsi_set_sense((*qc).dev, scmd, 5u8, 33u8, 0u8);
// "Logical Block Address out of range"
return 1;
}
// treat all other errors as -EINVAL, fall through
}
}
}
ata_scsi_set_invalid_field((*qc).dev, scmd, fp, 255u8);
return 1;
}
}

fn ata_scsi_qc_done(qc: *mut ata_queued_cmd, set_result: bool, scmd_result: u32) {
let cmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let done: Option<unsafe extern "C" fn(*mut scsi_cmnd)> = unsafe { (*qc).scsidone };
unsafe { ata_qc_free(qc) };
if set_result {
unsafe {
(*cmd).result = scmd_result as i32;
}
}
unsafe { done.unwrap()(cmd) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_deferred_qc_work")]
extern "C" fn ata_scsi_deferred_qc_work(work: *mut work_struct) {
let link: *mut ata_link = container_of__97!(work, ata_link);
let ap: *mut ata_port = unsafe { (*link).ap };
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
// If we still have a deferred qc and we are not in EH, issue it. In
// such case, we should not need any more deferring the qc, so warn if
// qc_defer() says otherwise.
let qc: *mut ata_queued_cmd = unsafe { (*link).deferred_qc };
let __v4: bool = !qc.is_null() && !ata_port_eh_scheduled(ap);
if __v4 {
let mut __v5: i64;
let __ret_warn_on: i32 = ((unsafe { unsafe { (*(*ap).ops).qc_defer }.unwrap()(qc) }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("695: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 695b - ., 3; .popsection\n# {0}", const 695 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_scsi_deferred_qc_work_6, sym __slate_asm_string_ata_scsi_deferred_qc_work_7, const 1686 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v5 = (__ret_warn_on != 0) as i64;
unsafe {
(*link).deferred_qc = core::ptr::null_mut::<ata_queued_cmd>();
}
unsafe { ata_qc_issue(ap, qc) };
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
}

#[unsafe(no_mangle)]
extern "C" fn ata_scsi_requeue_deferred_qc(ap: *mut ata_port, timedout_scmd: *mut scsi_cmnd) -> u32 {
let mut action: u32 = 2;
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
let mut host_byte: u32 = 0;
lockdep_assert_held__2!(unsafe { (*ap).lock });
// If we have deferred QCs when a reset, a timeout or an NCQ command
// fails, do not try to be smart about what to do with the deferred
// commands and simply terminate them and let the SCSI layer decide
// what to do.
link = unsafe { ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 2u32) };
while !link.is_null() {
let qc: *mut ata_queued_cmd = unsafe { (*link).deferred_qc };
if !qc.is_null() {
// Clear the deferred QC so that the deferred work does not try
// to issue it.
unsafe {
(*link).deferred_qc = core::ptr::null_mut::<ata_queued_cmd>();
}
unsafe { cancel_work(core::ptr::addr_of_mut!((*link).deferred_qc_work)) };
// We are going to complete some scsi command, either with
// DID_TIME_OUT if the command timed out while waiting for being
// issued, or with DID_REQUEUE if another command timed out or
// we had a failed command. However, the block layer may re-issue
// these commands immediately, keeping the scsi host busy and
// thus preventing the SCSI EH task from running.
// So schedule EH on the port to prevent accepting new commands
// until everything is sorted out with the error or timeout that
// got us here in the first place. Note that we set EH pending
// on the port before calling ata_port_schedule_eh() so that we
// do not reenter this function from ata_eh_set_pending() with
// timedout_scmd being NULL and erroneously retry deferred QCs
// that have timed out on other links.
if !ata_port_eh_scheduled(ap) {
unsafe {
(*ap).pflags = (*ap).pflags | 1;
}
unsafe { ata_port_schedule_eh(ap) };
}
// If we are being called from scsi_timeout(), then we have a
// non-NULL timedout_scmd. If the timed out command is for a
// deferred QC, terminate that deferred QC with DID_TIME_OUT and
// requeue all other deferred QCs. In this case we need to
// return SCSI_EH_DONE, because the timed out command was
// handled.
// If the timed out command is not for a deferred QC, we need to
// requeue all deferred QCs, and return SCSI_EH_NOT_HANDLED so
// that the timed out command gets added to the EH work queue
// with scsi_eh_scmd_add(), for later handling with libata EH
// ata_scsi_cmd_error_handler().
// If timedout_scmd is NULL, we simply need to requeue all
// deferred QCs and the return value does not matter as we were
// not called from scsi_timeout().
if !timedout_scmd.is_null() && (unsafe { (*qc).scsicmd }) == timedout_scmd {
host_byte = 3;
action = 0;
} else {
host_byte = 13;
}
ata_scsi_qc_done(qc, true, host_byte << 16i32);
}
link = unsafe { ata_link_next(link, ap, 2u32) };
}
action
}

fn ata_scsi_schedule_deferred_qc(link: *mut ata_link) {
let qc: *mut ata_queued_cmd = unsafe { (*link).deferred_qc };
let ap: *mut ata_port = unsafe { (*link).ap };
lockdep_assert_held__2!(unsafe { (*ap).lock });
// If we have a deferred qc, then qc_defer() is defined and we can use
// this callback to determine if this qc is good to go, unless EH has
// been scheduled.
if qc.is_null() {
return;
}
if ata_port_eh_scheduled(ap) {
ata_scsi_requeue_deferred_qc(ap, core::ptr::null_mut::<scsi_cmnd>());
return;
}
if (unsafe { unsafe { (*(*ap).ops).qc_defer }.unwrap()(qc) }) == 0 {
queue_work(unsafe { system_highpri_wq }, unsafe { core::ptr::addr_of_mut!((*link).deferred_qc_work) });
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_retry_deferred_qc")]
extern "C" fn ata_scsi_retry_deferred_qc(ap: *mut ata_port, scmd: *mut scsi_cmnd) -> u32 {
let mut action: u32 = 0;
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
action = ata_scsi_requeue_deferred_qc(ap, scmd);
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
action
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_retry_deferred_qc_697: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ata_port, *mut scsi_cmnd) -> u32>, *mut ()>(Some(ata_scsi_retry_deferred_qc)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_eh_timed_out")]
extern "C" fn ata_scsi_eh_timed_out(scmd: *mut scsi_cmnd) -> u32 {
let ap: *mut ata_port = ata_shost_to_port(unsafe { (*(*scmd).device).host });
// ata_scsi_cmd_error_handler() takes care of commands that timed out
// while executing. However, if we have deferred QCs while a timeout
// triggers, we must requeue these commands for retry so that we do not
// unnecessarily delay starting the SCSI EH task until these deferred
// commands also time out.
ata_scsi_retry_deferred_qc(ap, scmd)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_eh_timed_out_698: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_cmnd) -> u32>, *mut ()>(Some(ata_scsi_eh_timed_out)) };

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_qc_complete")]
extern "C" fn ata_scsi_qc_complete(qc: *mut ata_queued_cmd) {
let link: *mut ata_link = unsafe { (*(*qc).dev).link };
let cmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let cdb: *mut u8 = unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 };
let have_sense: bool = (unsafe { (*qc).flags }) & 131072 != 0;
let is_ata_passthru: bool = ((unsafe { *cdb.wrapping_offset(0 as isize) }) as i32) == 133 || ((unsafe { *cdb.wrapping_offset(0 as isize) }) as i32) == 161;
let is_ck_cond_request: bool = ((unsafe { *cdb.wrapping_offset(2 as isize) }) as i32) & 32 != 0;
let is_error: bool = (unsafe { (*qc).err_mask }) != 0;
// For ATA pass thru (SAT) commands, generate a sense block if
// user mandated it or if there's an error.  Note that if we
// generate because the user forced us to [CK_COND=1], a check
// condition is generated and the ATA register values are returned
// whether the command completed successfully or not. If there
// was no error, and CK_COND=1, we use the following sense data:
// sk = RECOVERED ERROR
// asc,ascq = ATA PASS-THROUGH INFORMATION AVAILABLE
if is_ata_passthru && (is_ck_cond_request || is_error || have_sense) {
if !have_sense {
ata_gen_passthru_sense(qc);
}
ata_scsi_set_passthru_sense_fields(qc);
if is_ck_cond_request {
set_status_byte(unsafe { (*qc).scsicmd }, 2u8);
}
} else {
if is_error {
if !have_sense {
ata_gen_ata_sense(qc);
}
ata_scsi_set_sense_information(qc);
}
}
ata_scsi_qc_done(qc, false, 0u32);
ata_scsi_schedule_deferred_qc(link);
}

fn ata_scsi_qc_issue(ap: *mut ata_port, qc: *mut ata_queued_cmd) -> i32 {
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
unsafe {
'__join_3: {
link = (*(*qc).dev).link;
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (*(*ap).ops).qc_defer != None {
// If we already have a deferred qc, then rely on the SCSI layer to
// requeue and defer all incoming commands until the deferred qc is
// processed, once all on-going commands complete.
if !unsafe { (*link).deferred_qc }.is_null() {
ata_qc_free(qc);
return 4182;
} else {
// Check if the command needs to be deferred.
*__slate_slot_3 = unsafe { (*(*ap).ops).qc_defer }.unwrap()(qc);
let __t0: i32 = *__slate_slot_3;
if __t0 == 0 {
break '__join_3;
} else {
if __t0 == 1 {
*__slate_slot_3 = 4182;
if !ata_is_ncq((*qc).tf.protocol) {
(*link).deferred_qc = qc;
return 0;
}
// We must defer this qc: if this is not an NCQ command, keep
// this qc as a deferred one and report to the SCSI layer that
// we issued it so that it is not requeued. The deferred qc will
// be issued with the port deferred_qc_work once all on-going
// commands complete.
} else {
if __t0 == 3 {
*__slate_slot_3 = 4182;
// Drivers making use of ap->excl_link cannot store the QC in
// link->deferred_qc, because the ap->excl_link handling is
// incompatible with the link->deferred_qc workqueue handling.
} else {
if __t0 == 2 {
*__slate_slot_3 = 4181;
} else {
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("699: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 699b - ., 3; .popsection\n# {0}", const 699 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_scsi_qc_issue_5, sym __slate_asm_string_ata_scsi_qc_issue_6, const 1897 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("700: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 700b - ., 4; .popsection\n# {0}", const 700 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
*__slate_slot_3 = 4181;
}
}
}
}
ata_qc_free(qc); // Force a requeue of the command to defer its execution.
return *__slate_slot_3;
}
}
}
ata_qc_issue(ap, qc);
return 0;
}
}

/// ata_scsi_translate - Translate then issue SCSI command to ATA device
/// @dev: ATA device to which the command is addressed
/// @cmd: SCSI command to execute
/// @xlat_func: Actor which translates @cmd to an ATA taskfile
/// @ap: ATA port of interest
///
/// Our ->queuecommand() function has decided that the SCSI
/// command issued can be directly translated into an ATA
/// command, rather than handled internally.
///
/// This function sets up an ata_queued_cmd structure for the
/// SCSI command, and sends that ata_queued_cmd to the hardware.
///
/// The xlat_func argument (actor) returns 0 if ready to execute
/// ATA command, else 1 to finish translation. If 1 is returned
/// then cmd->result (and possibly cmd->sense_buffer) are assumed
/// to be set reflecting an error condition or clean (early)
/// termination.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// 0 on success, SCSI_ML_QUEUE_DEVICE_BUSY or SCSI_MLQUEUE_HOST_BUSY if the
/// command needs to be deferred.
fn ata_scsi_translate(dev: *mut ata_device, cmd: *mut scsi_cmnd, xlat_func: Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>, ap: *mut ata_port) -> i32 {
let mut qc: core::mem::MaybeUninit<*mut ata_queued_cmd> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut ata_queued_cmd = core::ptr::addr_of_mut!(qc) as *mut *mut ata_queued_cmd;
unsafe {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
lockdep_assert_held__2!(unsafe { (*ap).lock });
// ata_scsi_qc_new() calls scsi_done(cmd) in case of failure. So we
// have nothing further to do when allocating a qc fails.
*__slate_slot_1 = ata_scsi_qc_new(dev, cmd);
if (*__slate_slot_1).is_null() {
return 0;
} else {
'__join_0: {
// data is present; dma-map it
if ((*cmd).sc_data_direction as u32) == 2 || ((*cmd).sc_data_direction as u32) == 1 {
if unlikely!(scsi_bufflen(cmd) < 1) != 0 {
while false {
}
_printk(c"\x014ata%u.%02u: WARNING: zero len r/w req\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno));
(*cmd).result = 458752i32;
break '__join_0;
} else {
ata_sg_init(*__slate_slot_1, scsi_sglist(cmd), scsi_sg_count(cmd));
(*(*__slate_slot_1)).dma_dir = ((*cmd).sc_data_direction as u32) as i32;
}
}
(*(*__slate_slot_1)).complete_fn = Some(ata_scsi_qc_complete);
if xlat_func.unwrap()(*__slate_slot_1) == 0 {
return ata_scsi_qc_issue(ap, *__slate_slot_1);
}
}
ata_qc_free(*__slate_slot_1);
scsi_done(cmd);
return 0;
}
}
}

/// ata_scsi_rbuf_fill - wrapper for SCSI command simulators
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @actor: Callback hook for desired SCSI command simulator
///
/// Takes care of the hard work of simulating a SCSI command...
/// Mapping the response buffer, calling the command's handler,
/// and handling the handler's return value.  This return value
/// indicates whether the handler wishes the SCSI command to be
/// completed successfully (0), or not (in which case cmd->result
/// and sense buffer are assumed to be set).
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsi_rbuf_fill(dev: *mut ata_device, cmd: *mut scsi_cmnd, actor: Option<unsafe extern "C" fn(*mut ata_device, *mut scsi_cmnd, *mut u8) -> u32>) {
let mut flags: u64 = 0;
let mut len: u32 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(ata_scsi_rbuf_lock) }, flags);
unsafe { memset((core::ptr::addr_of_mut!(ata_scsi_rbuf.0) as *mut u8) as *mut (), 0i32, 2048u64) };
len = unsafe { actor.unwrap()(dev, cmd, core::ptr::addr_of_mut!(ata_scsi_rbuf.0) as *mut u8) };
if len != 0 {
let mut __v3: i64;
let __ret_warn_on: i32 = (len > 2048) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("701: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 701b - ., 3; .popsection\n# {0}", const 701 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_scsi_rbuf_fill_4, sym __slate_asm_string_ata_scsi_rbuf_fill_5, const 2023 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("702: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 702b - ., 4; .popsection\n# {0}", const 702 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
if __v3 != 0 {
ata_scsi_set_sense(dev, cmd, 11u8, 0u8, 0u8);
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(ata_scsi_rbuf_lock) }, flags);
return;
}
unsafe { sg_copy_from_buffer(scsi_sglist(cmd), scsi_sg_count(cmd), (core::ptr::addr_of_mut!(ata_scsi_rbuf.0) as *mut u8) as *const (), len as u64) };
unsafe {
(*cmd).result = 0;
}
if scsi_bufflen(cmd) > len {
scsi_set_resid(cmd, scsi_bufflen(cmd).wrapping_sub(len));
}
}
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!(ata_scsi_rbuf_lock) }, flags);
}

/// ata_scsiop_inq_std - Simulate standard INQUIRY command
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Returns standard device identification data associated
/// with non-VPD INQUIRY command output.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_std(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
// SAM-3 (no version claimed)
// SBC-2 (no version claimed)
// SPC-3 (no version claimed)
// SAM-5 (no version claimed)
// SBC-4 (no version claimed)
// SPC-5 (no version claimed)
// ZBC r05
let mut hdr__0: [u8; 8] = [0, 0, 5, 2, 91u8, 0, 0, 2]; // claim SPC-3 version compatibility
// Set the SCSI Removable Media Bit (RMB) if the ATA removable media
// device bit (obsolete since ATA-8 ACS) is set.
if ata_id_removable!((*dev).__slate_anon_26.id) != 0 {
let __v0: *mut u8 = (hdr__0.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 128i32) as u8;
}
}
if (unsafe { (*dev).class }) == 9 {
hdr__0[0] = 20;
hdr__0[2] = 7; // claim SPC-5 version compatibility
}
if (unsafe { (*dev).flags }) & 8192 != 0 {
hdr__0[2] = 13;
}
// claim SPC-6 version compatibility
unsafe { memcpy(rbuf as *mut (), hdr__0.as_mut_ptr() as *const (), core::mem::size_of::<[u8; 8]>() as u64) };
unsafe { memcpy(rbuf.wrapping_offset(8 as isize) as *mut (), b"ATA     \0".as_ptr() as *const (), 8u64) };
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, rbuf.wrapping_offset(16 as isize), 27u32, 16u32) };
// From SAT, use last 2 words from fw rev unless they are spaces
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, rbuf.wrapping_offset(32 as isize), 25u32, 4u32) };
if (unsafe { strncmp(rbuf.wrapping_offset(32 as isize) as *const u8, c"    ".as_ptr() as *const u8, 4u64) }) == 0 {
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, rbuf.wrapping_offset(32 as isize), 23u32, 4u32) };
}
if ((unsafe { *rbuf.wrapping_offset(32 as isize) }) as i32) == 0 || ((unsafe { *rbuf.wrapping_offset(32 as isize) }) as i32) == 32 {
unsafe { memcpy(rbuf.wrapping_offset(32 as isize) as *mut (), b"n/a \0".as_ptr() as *const (), 4u64) };
}
if ata_dev_is_zoned(dev) {
unsafe { memcpy(rbuf.wrapping_offset(58 as isize) as *mut (), (core::ptr::addr_of!(versions_zbc) as *const u8) as *const (), core::mem::size_of::<[u8; 8]>() as u64) };
} else {
unsafe { memcpy(rbuf.wrapping_offset(58 as isize) as *mut (), (core::ptr::addr_of!(versions) as *const u8) as *const (), core::mem::size_of::<[u8; 6]>() as u64) };
}
// Include all 8 possible version descriptors, even if not all of
// them are popoulated.
96
}

static versions: [u8; 6] = [0, 96, 3, 32, 3, 0];

static versions_zbc: [u8; 8] = [0, 160, 6, 0, 5, 192, 96, 36];

/// ata_scsiop_inq_00 - Simulate INQUIRY VPD page 0, list of pages
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Returns list of inquiry VPD pages available.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_00(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let mut i: i32 = 0;
let mut num_pages: i32 = 0; // page 0x00, this page
// page 0x80, unit serial no page
// page 0x83, device ident page
// page 0x89, ata info page
// page 0xb0, block limits page
// page 0xb1, block device characteristics page
// page 0xb2, thin provisioning page
// page 0xb6, zoned block device characteristics
// page 0xb9, concurrent positioning ranges
while (i as u64) < (core::mem::size_of::<[u8; 9]>() as u64) {
let __v0: bool = ((unsafe { pages[i as usize] }) as i32) == 182 && !ata_dev_is_zoned(dev);
if !__v0 {
unsafe {
*rbuf.wrapping_offset(num_pages.wrapping_add(4) as isize) = pages[i as usize];
}
num_pages = num_pages.wrapping_add(1);
}
i = i.wrapping_add(1);
}
unsafe {
*rbuf.wrapping_offset(3 as isize) = num_pages as u8;
}
// number of supported VPD pages
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

static pages: [u8; 9] = [0, 128, 131, 137, 176, 177, 178, 182, 185];

/// ata_scsiop_inq_80 - Simulate INQUIRY VPD page 80, device serial number
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Returns ATA device serial number.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_80(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
// this page code
// page len
unsafe { memcpy(rbuf as *mut (), (core::ptr::addr_of!(hdr) as *const u8) as *const (), core::mem::size_of::<[u8; 4]>() as u64) };
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, rbuf.wrapping_offset(4 as isize), 10u32, 20u32) };
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

static hdr: [u8; 4] = [0, 128, 0, 20];

/// ata_scsiop_inq_83 - Simulate INQUIRY VPD page 83, device identity
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Yields two logical unit device identification designators:
///  - vendor specific ASCII containing the ATA serial number
///  - SAT defined "t10 vendor id based" containing ASCII vendor
///    name ("ATA     "), model and serial numbers.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_83(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let sat_model_serial_desc_len: i32 = 68;
let mut num: i32 = 4;
unsafe {
*rbuf.wrapping_offset(1 as isize) = 131;
}
// this page code
// piv=0, assoc=lu, code_set=ACSII, designator=vendor
unsafe {
*rbuf.wrapping_offset(num.wrapping_add(0) as isize) = 2;
}
unsafe {
*rbuf.wrapping_offset(num.wrapping_add(3) as isize) = 20;
}
num = num.wrapping_add(4);
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, rbuf.wrapping_offset(num as isize), 10u32, 20u32) };
num = num.wrapping_add(20);
// SAT defined lu model and serial numbers descriptor
//
// piv=0, assoc=lu, code_set=ACSII, designator=t10 vendor id
unsafe {
*rbuf.wrapping_offset(num.wrapping_add(0) as isize) = 2;
}
unsafe {
*rbuf.wrapping_offset(num.wrapping_add(1) as isize) = 1;
}
unsafe {
*rbuf.wrapping_offset(num.wrapping_add(3) as isize) = sat_model_serial_desc_len as u8;
}
num = num.wrapping_add(4);
unsafe { memcpy(rbuf.wrapping_offset(num as isize) as *mut (), b"ATA     \0".as_ptr() as *const (), 8u64) };
num = num.wrapping_add(8);
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, rbuf.wrapping_offset(num as isize), 27u32, 40u32) };
num = num.wrapping_add(40);
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, rbuf.wrapping_offset(num as isize), 10u32, 20u32) };
num = num.wrapping_add(20);
if ata_id_has_wwn((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16) {
// SAT defined lu world wide name
//
// piv=0, assoc=lu, code_set=binary, designator=NAA
unsafe {
*rbuf.wrapping_offset(num.wrapping_add(0) as isize) = 1;
}
unsafe {
*rbuf.wrapping_offset(num.wrapping_add(1) as isize) = 3;
}
unsafe {
*rbuf.wrapping_offset(num.wrapping_add(3) as isize) = 8;
}
num = num.wrapping_add(4);
unsafe { ata_id_string(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16, rbuf.wrapping_offset(num as isize), 108u32, 8u32) };
num = num.wrapping_add(8);
}
unsafe {
*rbuf.wrapping_offset(3 as isize) = num.wrapping_sub(4) as u8;
}
// page len (assume less than 256 bytes)
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

/// ata_scsiop_inq_89 - Simulate INQUIRY VPD page 89, ATA info
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Yields SAT-specified ATA VPD page.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_89(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
unsafe {
*rbuf.wrapping_offset(1 as isize) = 137;
}
// our page code
unsafe {
*rbuf.wrapping_offset(2 as isize) = 2u8;
}
// page size fixed at 238h
unsafe {
*rbuf.wrapping_offset(3 as isize) = 56u8;
}
unsafe { memcpy(rbuf.wrapping_offset(8 as isize) as *mut (), b"linux   \0".as_ptr() as *const (), 8u64) };
unsafe { memcpy(rbuf.wrapping_offset(16 as isize) as *mut (), b"libata          \0".as_ptr() as *const (), 16u64) };
unsafe { memcpy(rbuf.wrapping_offset(32 as isize) as *mut (), b"3.00\0".as_ptr() as *const (), 4u64) };
unsafe {
*rbuf.wrapping_offset(36 as isize) = 52;
}
// force D2H Reg FIS (34h)
unsafe {
*rbuf.wrapping_offset(37 as isize) = 128u8;
}
// bit 7 indicates Command FIS
// TODO: PMP?
// we don't store the ATA device signature, so we fake it
unsafe {
*rbuf.wrapping_offset(38 as isize) = 64;
}
// really, this is Status reg
unsafe {
*rbuf.wrapping_offset(40 as isize) = 1;
}
unsafe {
*rbuf.wrapping_offset(48 as isize) = 1;
}
unsafe {
*rbuf.wrapping_offset(56 as isize) = 236;
}
unsafe { memcpy(rbuf.wrapping_offset(60 as isize) as *mut (), ((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16).wrapping_offset(0 as isize) as *const (), 512u64) };
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

// ata_dsm_trim_pages - maximum DSM TRIM payload for a device, in 512-byte pages
// @dev: ATA device the DATA SET MANAGEMENT TRIM command will be sent to
//
// A DATA SET MANAGEMENT TRIM payload is a list of 512-byte pages, each holding
// up to ATA_MAX_TRIM_RNUM (64) LBA Range Entries; the format is page-based and
// unrelated to the logical sector size.
//
// The logical sector size still bounds it, though: the descriptor is written
// directly into the WRITE SAME data-out buffer, which sd sizes to a single
// logical block, so it can hold at most sector_size / 512 pages.
//
// Return: the maximum number of 512-byte pages a single translated WRITE SAME
// command may send to @dev, that is the smaller of MAX PAGES PER DSM COMMAND
// (IDENTIFY DEVICE word 105, when the device reports a non-zero limit) and
// the logical sector size expressed in 512-byte pages; never less than one.
fn ata_dsm_trim_pages(dev: *mut ata_device) -> u32 {
let sector_size: u32 = ata_id_logical_sector_size((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16);
let mut max_pages: u32 = ata_id_dsm_max_pages((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16) as u32;
let mut pages__0: u32 = sector_size / 512;
// If the device does not specify a limit, assume only a single page.
if max_pages == 0 {
max_pages = 1;
}
let mut __v0: u32;
let __x: u32 = pages__0;
let __y: u32 = max_pages;
let __v1: u32 = if __x == 0 {
__y
} else {
let __v2: u32 = if __y == 0 {
__x
} else {
let mut __v3: u32;
let __UNIQUE_ID_x__703: u32 = __x;
let __UNIQUE_ID_y__704: u32 = __y;
loop {
if __types_ok__2!(__UNIQUE_ID_x__703, __UNIQUE_ID_y__704) == 0 {
unsafe { __compiletime_assert_705() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v3 = if __UNIQUE_ID_x__703 < __UNIQUE_ID_y__704 { __UNIQUE_ID_x__703 } else { __UNIQUE_ID_y__704 };
__v3
};
__v2
};
__v0 = __v1;
pages__0 = __v0;
pages__0
}

/// ata_scsiop_inq_b0 - Simulate INQUIRY VPD page B0, Block Limits
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Return data for the VPD page B0h (Block Limits).
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_b0(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let mut min_io_sectors: u16 = 0;
unsafe {
*rbuf.wrapping_offset(1 as isize) = 176;
}
unsafe {
*rbuf.wrapping_offset(3 as isize) = 60;
}
// required VPD size with unmap support
// Optimal transfer length granularity.
//
// This is always one physical block, but for disks with a smaller
// logical than physical sector size we need to figure out what the
// latter is.
min_io_sectors = (1i32 << (ata_id_log2_per_physical_sector((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16) as i32)) as u16;
put_unaligned_be16(min_io_sectors, rbuf.wrapping_offset(6 as isize) as *mut ());
// Optimal unmap granularity.
//
// The ATA spec doesn't even know about a granularity or alignment
// for the TRIM command.  We can leave away most of the unmap related
// VPD page entries, but we have specifify a granularity to signal
// that we support some form of unmap - in thise case via WRITE SAME
// with the unmap bit set.
if ata_id_has_trim((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16) {
let max_pages: u32 = ata_dsm_trim_pages(dev);
let mut max_blocks: u64 = (max_pages.wrapping_mul(64) as u64).wrapping_mul(65535u64);
if (unsafe { (*dev).quirks }) & 16777216 != 0 {
max_blocks = 262144u64;
}
put_unaligned_be64(max_blocks, rbuf.wrapping_offset(36 as isize) as *mut ());
put_unaligned_be32(1u32, rbuf.wrapping_offset(28 as isize) as *mut ());
}
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

/// ata_scsiop_inq_b1 - Simulate INQUIRY VPD page B1, Block Device
/// 		    Characteristics
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Return data for the VPD page B1h (Block Device Characteristics).
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_b1(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let form_factor: i32 = ata_id_form_factor((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16);
let media_rotation_rate: i32 = ata_id_rotation_rate((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16);
let zoned: u8 = ata_id_zoned_cap((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16);
unsafe {
*rbuf.wrapping_offset(1 as isize) = 177;
}
unsafe {
*rbuf.wrapping_offset(3 as isize) = 60;
}
unsafe {
*rbuf.wrapping_offset(4 as isize) = (media_rotation_rate >> 8i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(5 as isize) = media_rotation_rate as u8;
}
unsafe {
*rbuf.wrapping_offset(7 as isize) = form_factor as u8;
}
if zoned != 0 {
unsafe {
*rbuf.wrapping_offset(8 as isize) = ((zoned as i32) << 4i32) as u8;
}
}
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

/// ata_scsiop_inq_b2 - Simulate INQUIRY VPD page B2, Logical Block
/// 		    Provisioning
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Return data for the VPD page B2h (Logical Block Provisioning).
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_b2(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
// SCSI Thin Provisioning VPD page: SBC-3 rev 22 or later
unsafe {
*rbuf.wrapping_offset(1 as isize) = 178;
}
unsafe {
*rbuf.wrapping_offset(3 as isize) = 4;
}
unsafe {
*rbuf.wrapping_offset(5 as isize) = 64u8;
}
// TPWS
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

/// ata_scsiop_inq_b6 - Simulate INQUIRY VPD page B6, Zoned Block Device
/// 		    Characteristics
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Return data for the VPD page B2h (Zoned Block Device Characteristics).
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_b6(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
if !ata_dev_is_zoned(dev) {
ata_scsi_set_invalid_field(dev, cmd, 2u16, 255u8);
return 0;
}
// zbc-r05 SCSI Zoned Block device characteristics VPD page
unsafe {
*rbuf.wrapping_offset(1 as isize) = 182;
}
unsafe {
*rbuf.wrapping_offset(3 as isize) = 60;
}
// URSWRZ bit is only meaningful for host-managed ZAC drives
if (unsafe { (*dev).zac_zoned_cap }) & 1 != 0 {
let __v0: *mut u8 = rbuf.wrapping_offset(4 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 1) as u8;
}
}
put_unaligned_be32(unsafe { (*dev).zac_zones_optimal_open }, rbuf.wrapping_offset(8 as isize) as *mut ());
put_unaligned_be32(unsafe { (*dev).zac_zones_optimal_nonseq }, rbuf.wrapping_offset(12 as isize) as *mut ());
put_unaligned_be32(unsafe { (*dev).zac_zones_max_open }, rbuf.wrapping_offset(16 as isize) as *mut ());
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

/// ata_scsiop_inq_b9 - Simulate INQUIRY VPD page B9, Concurrent Positioning
/// 		    Ranges
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Return data for the VPD page B9h (Concurrent Positioning Ranges).
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsiop_inq_b9(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let cpr_log: *mut ata_cpr_log = unsafe { (*dev).cpr_log };
let mut desc: *mut u8 = rbuf.wrapping_offset(64 as isize);
let mut i: i32 = 0;
if cpr_log.is_null() {
ata_scsi_set_invalid_field(dev, cmd, 2u16, 255u8);
return 0;
}
// SCSI Concurrent Positioning Ranges VPD page: SBC-5 rev 1 or later
unsafe {
*rbuf.wrapping_offset(1 as isize) = 185;
}
put_unaligned_be16(64i32.wrapping_add(((unsafe { (*cpr_log).nr_cpr }) as i32).wrapping_mul(32)).wrapping_sub(4) as u16, rbuf.wrapping_offset(2 as isize) as *mut ());
while i < ((unsafe { (*cpr_log).nr_cpr }) as i32) {
unsafe {
*desc.wrapping_offset(0 as isize) = (*(core::ptr::addr_of_mut!((*cpr_log).cpr) as *mut ata_cpr).wrapping_offset(i as isize)).num;
}
unsafe {
*desc.wrapping_offset(1 as isize) = (*(core::ptr::addr_of_mut!((*cpr_log).cpr) as *mut ata_cpr).wrapping_offset(i as isize)).num_storage_elements;
}
put_unaligned_be64(unsafe { (*(core::ptr::addr_of_mut!((*cpr_log).cpr) as *mut ata_cpr).wrapping_offset(i as isize)).start_lba }, desc.wrapping_offset(8 as isize) as *mut ());
put_unaligned_be64(unsafe { (*(core::ptr::addr_of_mut!((*cpr_log).cpr) as *mut ata_cpr).wrapping_offset(i as isize)).num_lbas }, desc.wrapping_offset(16 as isize) as *mut ());
i = i.wrapping_add(1);
let __v3: *mut u8 = desc;
desc = __v3.wrapping_offset(32 as isize);
}
(get_unaligned_be16(rbuf.wrapping_offset(2 as isize) as *const ()) as i32).wrapping_add(4) as u32
}

/// ata_scsiop_inquiry - Simulate INQUIRY command
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Returns data associated with an INQUIRY command output.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsiop_inquiry")]
extern "C" fn ata_scsiop_inquiry(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let scsicmd: *const u8 = (unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }) as *const u8;
// is CmdDt set?
if ((unsafe { *scsicmd.wrapping_offset(1 as isize) }) as i32) & 2 != 0 {
ata_scsi_set_invalid_field(dev, cmd, 1u16, 255u8);
return 0;
}
// Is EVPD clear?
if ((unsafe { *scsicmd.wrapping_offset(1 as isize) }) as i32) & 1 == 0 {
return ata_scsiop_inq_std(dev, cmd, rbuf);
}
match (unsafe { *scsicmd.wrapping_offset(2 as isize) }) as i32 {
0 => {
return ata_scsiop_inq_00(dev, cmd, rbuf);
}
128 => {
return ata_scsiop_inq_80(dev, cmd, rbuf);
}
131 => {
return ata_scsiop_inq_83(dev, cmd, rbuf);
}
137 => {
return ata_scsiop_inq_89(dev, cmd, rbuf);
}
176 => {
return ata_scsiop_inq_b0(dev, cmd, rbuf);
}
177 => {
return ata_scsiop_inq_b1(dev, cmd, rbuf);
}
178 => {
return ata_scsiop_inq_b2(dev, cmd, rbuf);
}
182 => {
return ata_scsiop_inq_b6(dev, cmd, rbuf);
}
185 => {
return ata_scsiop_inq_b9(dev, cmd, rbuf);
}
_ => {
ata_scsi_set_invalid_field(dev, cmd, 2u16, 255u8);
return 0;
}
}
unsafe { core::mem::zeroed() }
}

/// modecpy - Prepare response for MODE SENSE
/// @dest: output buffer
/// @src: data being copied
/// @n: length of mode page
/// @changeable: whether changeable parameters are requested
///
/// Generate a generic MODE SENSE page for either current or changeable
/// parameters.
///
/// LOCKING:
/// None.
fn modecpy(dest: *mut u8, src: *const u8, n: i32, changeable: bool) {
if changeable {
unsafe { memcpy(dest as *mut (), src as *const (), 2u64) };
unsafe { memset(dest.wrapping_offset(2 as isize) as *mut (), 0i32, n.wrapping_sub(2) as u64) };
} else {
unsafe { memcpy(dest as *mut (), src as *const (), n as u64) };
}
}

/// ata_msense_caching - Simulate MODE SENSE caching info page
/// @id: device IDENTIFY data
/// @buf: output buffer
/// @changeable: whether changeable parameters are requested
///
/// Generate a caching info page, which conditionally indicates
/// write caching to the SCSI layer, depending on device
/// capabilities.
///
/// LOCKING:
/// None.
fn ata_msense_caching(id: *mut u16, buf: *mut u8, changeable: bool) -> u32 {
modecpy(buf, unsafe { core::ptr::addr_of!(def_cache_mpage.0) as *const u8 }, core::mem::size_of::<[u8; 20]>() as i32, changeable);
if changeable {
let __v0: *mut u8 = buf.wrapping_offset(2 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 4i32) as u8;
}
// ata_mselect_caching()
} else {
let __v3: i32 = (ata_id_wcache_enabled(id as *const u16) as i32) << 2i32;
let __v4: *mut u8 = buf.wrapping_offset(2 as isize);
unsafe {
*__v4 = ((*__v4 as i32) | __v3) as u8;
}
// write cache enable
let __v7: i32 = (!ata_id_rahead_enabled(id as *const u16) as i32) << 5i32;
let __v8: *mut u8 = buf.wrapping_offset(12 as isize);
unsafe {
*__v8 = ((*__v8 as i32) | __v7) as u8;
}
// disable read ahead
}
core::mem::size_of::<[u8; 20]>() as u32
}

/// Simulate MODE SENSE control mode page, sub-page 0.
fn ata_msense_control_spg0(dev: *mut ata_device, buf: *mut u8, changeable: bool) -> u32 {
modecpy(buf, unsafe { core::ptr::addr_of!(def_control_mpage) as *const u8 }, core::mem::size_of::<[u8; 12]>() as i32, changeable);
if changeable {
// ata_mselect_control()
let __v0: *mut u8 = buf.wrapping_offset(2 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 4i32) as u8;
}
} else {
let d_sense: bool = (unsafe { (*dev).flags }) & 2147483648 != 0;
// descriptor format sense data
let __v3: *mut u8 = buf.wrapping_offset(2 as isize);
unsafe {
*__v3 = ((*__v3 as i32) | (d_sense as i32) << 2i32) as u8;
}
}
core::mem::size_of::<[u8; 12]>() as u32
}

/// Translate an ATA duration limit in microseconds to a SCSI duration limit
/// using the t2cdlunits 0xa (10ms). Since the SCSI duration limits are 2-bytes
/// only, take care of overflows.
fn ata_xlat_cdl_limit(buf: *mut u8) -> u16 {
let limit: u32 = get_unaligned_le32(buf as *const ());
min_t!(u32, limit / 10000, 65535i32) as u16
}

/// Simulate MODE SENSE control mode page, sub-pages 07h and 08h
/// (command duration limits T2A and T2B mode pages).
fn ata_msense_control_spgt2(dev: *mut ata_device, buf: *mut u8, spg: u8) -> u32 {
let mut b: *mut u8 = unsafe { core::mem::zeroed() };
let mut desc: *mut u8 = unsafe { core::mem::zeroed() };
let mut policy: u32 = 0;
let mut i: i32 = 0;
if (unsafe { (*dev).flags }) & 8192 == 0 || unsafe { (*dev).cdl }.is_null() {
return 0;
}
let cdl: *mut u8 = unsafe { (*(*dev).cdl).desc_log_buf.as_mut_ptr() as *mut u8 };
// Fill the subpage. The first four bytes of the T2A/T2B mode pages
// are a header. The PAGE LENGTH field is the size of the page
// excluding the header.
unsafe {
*buf.wrapping_offset(0 as isize) = 10;
}
unsafe {
*buf.wrapping_offset(1 as isize) = spg;
}
put_unaligned_be16(228u16, buf.wrapping_offset(2 as isize) as *mut ());
if (spg as i32) == 7 {
// Read descriptors map to the T2A page:
// set perf_vs_duration_guidleine.
unsafe {
*buf.wrapping_offset(7 as isize) = (((*cdl.wrapping_offset(0 as isize) as i32) & 3) << 4i32) as u8;
}
desc = cdl.wrapping_offset(64 as isize);
} else {
// Write descriptors map to the T2B page
desc = cdl.wrapping_offset(288 as isize);
}
// Fill the T2 page descriptors
b = buf.wrapping_offset(8 as isize);
policy = get_unaligned_le32(cdl.wrapping_offset(0 as isize) as *const ());
while i < 7 {
// t2cdlunits: fixed to 10ms
unsafe {
*b.wrapping_offset(0 as isize) = 10;
}
// Max inactive time and its policy
put_unaligned_be16(ata_xlat_cdl_limit(desc.wrapping_offset(8 as isize)), b.wrapping_offset(2 as isize) as *mut ());
unsafe {
*b.wrapping_offset(6 as isize) = ((policy >> 8i32 & 15) << 4i32) as u8;
}
// Max active time and its policy
put_unaligned_be16(ata_xlat_cdl_limit(desc.wrapping_offset(4 as isize)), b.wrapping_offset(4 as isize) as *mut ());
let __v0: *mut u8 = b.wrapping_offset(6 as isize);
unsafe {
*__v0 = (((*__v0 as i32) as u32) | policy >> 4i32 & 15) as u8;
}
// Command duration guideline and its policy
put_unaligned_be16(ata_xlat_cdl_limit(desc.wrapping_offset(16 as isize)), b.wrapping_offset(10 as isize) as *mut ());
unsafe {
*b.wrapping_offset(14 as isize) = (policy & 15) as u8;
}
i = i.wrapping_add(1);
let __v6: *mut u8 = b;
b = __v6.wrapping_offset(32 as isize);
let __v8: *mut u8 = desc;
desc = __v8.wrapping_offset(32 as isize);
}
232
}

/// Simulate MODE SENSE control mode page, sub-page f2h
/// (ATA feature control mode page).
fn ata_msense_control_ata_feature(dev: *mut ata_device, buf: *mut u8) -> u32 {
// PS=0, SPF=1
unsafe {
*buf.wrapping_offset(0 as isize) = 74u8;
}
unsafe {
*buf.wrapping_offset(1 as isize) = 242;
}
// The first four bytes of ATA Feature Control mode page are a header.
// The PAGE LENGTH field is the size of the page excluding the header.
put_unaligned_be16(12u16, buf.wrapping_offset(2 as isize) as *mut ());
if (unsafe { (*dev).flags }) & 16777216 != 0 {
unsafe {
*buf.wrapping_offset(4 as isize) = 2;
}
} else {
unsafe {
*buf.wrapping_offset(4 as isize) = 0;
}
}
// T2A and T2B pages enabled
16
}

/// ata_msense_control - Simulate MODE SENSE control mode page
/// @dev: ATA device of interest
/// @buf: output buffer
/// @spg: sub-page code
/// @changeable: whether changeable parameters are requested
///
/// Generate a generic MODE SENSE control mode page.
///
/// LOCKING:
/// None.
fn ata_msense_control(dev: *mut ata_device, buf: *mut u8, spg: u8, changeable: bool) -> u32 {
let mut n: u32 = 0;
match spg as i32 {
0 => {
return ata_msense_control_spg0(dev, buf, changeable);
}
7 | 8 => {
return ata_msense_control_spgt2(dev, buf, spg);
}
242 => {
return ata_msense_control_ata_feature(dev, buf);
}
255 => {
n = ata_msense_control_spg0(dev, buf, changeable);
let __v0: u32 = ata_msense_control_spgt2(dev, buf.wrapping_offset(n as isize), 7u8);
n = n.wrapping_add(__v0);
let __v3: u32 = ata_msense_control_spgt2(dev, buf.wrapping_offset(n as isize), 8u8);
n = n.wrapping_add(__v3);
let __v6: u32 = ata_msense_control_ata_feature(dev, buf.wrapping_offset(n as isize));
n = n.wrapping_add(__v6);
return n;
}
_ => {
return 0;
}
}
unsafe { core::mem::zeroed() }
}

/// ata_msense_rw_recovery - Simulate MODE SENSE r/w error recovery page
/// @buf: output buffer
/// @changeable: whether changeable parameters are requested
///
/// Generate a generic MODE SENSE r/w error recovery page.
///
/// LOCKING:
/// None.
fn ata_msense_rw_recovery(buf: *mut u8, changeable: bool) -> u32 {
modecpy(buf, unsafe { core::ptr::addr_of!(def_rw_recovery_mpage) as *const u8 }, core::mem::size_of::<[u8; 12]>() as i32, changeable);
core::mem::size_of::<[u8; 12]>() as u32
}

/// ata_scsiop_mode_sense - Simulate MODE SENSE 6, 10 commands
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Simulate MODE SENSE commands. Assume this is invoked for direct
/// access devices (e.g. disks) only. There should be no block
/// descriptor for other device types.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsiop_mode_sense")]
extern "C" fn ata_scsiop_mode_sense(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let mut fp: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut u16 = core::ptr::addr_of_mut!(fp) as *mut u16;
let mut bp: u8 = unsafe { core::mem::zeroed() };
let mut dpofua: u8 = unsafe { core::mem::zeroed() };
let mut six_byte: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut u32 = core::ptr::addr_of_mut!(six_byte) as *mut u32;
let mut page_control: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut u32 = core::ptr::addr_of_mut!(page_control) as *mut u32;
let mut ebd: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u32 = core::ptr::addr_of_mut!(ebd) as *mut u32;
let mut spg: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u8 = core::ptr::addr_of_mut!(spg) as *mut u8;
let mut pg: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut u8 = core::ptr::addr_of_mut!(pg) as *mut u8; // number of blocks: sat unspecified
// block length: 512 bytes
let mut p: *mut u8 = unsafe { core::mem::zeroed() };
let mut scsicmd: *mut u8 = unsafe { core::mem::zeroed() };
unsafe {
'__join_1: {
scsicmd = (*cmd).cmnd.as_mut_ptr() as *mut u8;
p = rbuf;
let __slate_auto_bytes = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
dpofua = 0;
bp = 255;
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
*__slate_slot_31 = ((*scsicmd.wrapping_offset(0 as isize) as i32) == 26) as u32;
*__slate_slot_33 = ((*scsicmd.wrapping_offset(1 as isize) as i32) & 8 == 0) as u32; // dbd bit inverted == edb
// LLBA bit in msense(10) ignored (compliant)
*__slate_slot_32 = ((*scsicmd.wrapping_offset(2 as isize) as i32) >> 6i32) as u32;
let __t2: u32 = *__slate_slot_32;
if __t2 != 0 {
if __t2 != 1 {
if __t2 != 2 {
if __t2 == 3 {
ata_scsi_set_sense(dev, cmd, 5u8, 57u8, 0u8);
// "Saving parameters not supported"
return 0;
} else {
*__slate_slot_28 = 2;
bp = 6;
break '__join_1;
}
}
}
}
if *__slate_slot_31 != 0 {
let __v25: *mut u8 = p;
p = __v25.wrapping_offset(4i32.wrapping_add(if *__slate_slot_33 != 0 { 8i32 } else { 0i32 }) as isize);
} else {
let __v27: *mut u8 = p;
p = __v27.wrapping_offset(8i32.wrapping_add(if *__slate_slot_33 != 0 { 8i32 } else { 0i32 }) as isize);
}
'__join_16: {
*__slate_slot_35 = ((*scsicmd.wrapping_offset(2 as isize) as i32) & 63) as u8;
*__slate_slot_34 = *scsicmd.wrapping_offset(3 as isize);
// Supported subpages: all subpages and sub-pages 07h, 08h and f2h of
// the control page.
if *__slate_slot_34 != 0 {
'__join_17: {
let __t1: i32 = *__slate_slot_34 as i32;
if __t1 == 255 {
break '__join_16;
} else {
if __t1 != 7 {
if __t1 != 8 {
if __t1 != 242 {
break '__join_17;
}
}
}
}
if (*dev).flags & 8192 != 0 && (*__slate_slot_35 as i32) == 10 {
break '__join_16;
}
}
*__slate_slot_28 = 3;
break '__join_1;
}
}
let __t0: i32 = *__slate_slot_35 as i32;
if __t0 == 1 {
let __v23: u32 = ata_msense_rw_recovery(p, *__slate_slot_32 == 1);
let __v22: *mut u8 = p;
p = __v22.wrapping_offset(__v23 as isize);
} else {
if __t0 == 8 {
let __v20: u32 = ata_msense_caching((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16, p, *__slate_slot_32 == 1);
let __v19: *mut u8 = p;
p = __v19.wrapping_offset(__v20 as isize);
} else {
if __t0 == 10 {
let __v17: u32 = ata_msense_control(dev, p, *__slate_slot_34, *__slate_slot_32 == 1);
let __v16: *mut u8 = p;
p = __v16.wrapping_offset(__v17 as isize);
} else {
if __t0 == 63 {
let __v14: u32 = ata_msense_rw_recovery(p, *__slate_slot_32 == 1);
let __v13: *mut u8 = p;
p = __v13.wrapping_offset(__v14 as isize);
let __v11: u32 = ata_msense_caching((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16, p, *__slate_slot_32 == 1);
let __v10: *mut u8 = p;
p = __v10.wrapping_offset(__v11 as isize);
let __v8: u32 = ata_msense_control(dev, p, *__slate_slot_34, *__slate_slot_32 == 1);
let __v7: *mut u8 = p;
p = __v7.wrapping_offset(__v8 as isize);
} else {
*__slate_slot_28 = 2; // invalid page code
break '__join_1;
}
}
}
}
if (*dev).flags & 512 != 0 {
dpofua = 16u8;
}
if *__slate_slot_31 != 0 {
*rbuf.wrapping_offset(0 as isize) = (p.offset_from(rbuf as *mut u8) as i64).wrapping_sub(1) as u8;
let __v5: *mut u8 = rbuf.wrapping_offset(2 as isize);
*__v5 = ((*__v5 as i32) | (dpofua as i32)) as u8;
if *__slate_slot_33 != 0 {
*rbuf.wrapping_offset(3 as isize) = core::mem::size_of::<[u8; 8]>() as u8;
memcpy(rbuf.wrapping_offset(4 as isize) as *mut (), (core::ptr::addr_of!(sat_blk_desc) as *const u8) as *const (), core::mem::size_of::<[u8; 8]>() as u64);
}
return (*rbuf.wrapping_offset(0 as isize) as i32).wrapping_add(1) as u32;
} else {
put_unaligned_be16((p.offset_from(rbuf as *mut u8) as i64).wrapping_sub(2) as u16, rbuf.wrapping_offset(0 as isize) as *mut ());
let __v2: *mut u8 = rbuf.wrapping_offset(3 as isize);
*__v2 = ((*__v2 as i32) | (dpofua as i32)) as u8;
if *__slate_slot_33 != 0 {
*rbuf.wrapping_offset(7 as isize) = core::mem::size_of::<[u8; 8]>() as u8;
memcpy(rbuf.wrapping_offset(8 as isize) as *mut (), (core::ptr::addr_of!(sat_blk_desc) as *const u8) as *const (), core::mem::size_of::<[u8; 8]>() as u64);
}
return (get_unaligned_be16(rbuf.wrapping_offset(0 as isize) as *const ()) as i32).wrapping_add(2) as u32;
}
}
ata_scsi_set_invalid_field(dev, cmd, *__slate_slot_28, bp);
return 0;
}
// current
// changeable
// defaults
// supported
// saved
}

static sat_blk_desc: [u8; 8] = [0, 0, 0, 0, 0, 0, 2, 0];

/// ata_scsiop_read_cap - Simulate READ CAPACITY[ 16] commands
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Simulate READ CAPACITY commands.
///
/// LOCKING:
/// None.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsiop_read_cap")]
extern "C" fn ata_scsiop_read_cap(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let scsicmd: *mut u8 = unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 };
let mut last_lba: u64 = unsafe { (*dev).n_sectors }.wrapping_sub(1); // LBA of the last block
let mut sector_size: u32 = 0; // physical sector size in bytes
let mut log2_per_phys: u8 = 0;
let mut lowest_aligned: u16 = 0;
sector_size = ata_id_logical_sector_size((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16);
log2_per_phys = ata_id_log2_per_physical_sector((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16);
lowest_aligned = ata_id_logical_sector_offset((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16, log2_per_phys);
if ((unsafe { *scsicmd.wrapping_offset(0 as isize) }) as i32) == 37 {
if last_lba >= 4294967295 {
last_lba = 4294967295;
}
// sector count, 32-bit
unsafe {
*rbuf.wrapping_offset(0 as isize) = (last_lba >> 24i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(1 as isize) = (last_lba >> 16i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(2 as isize) = (last_lba >> 8i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(3 as isize) = last_lba as u8;
}
// sector size
unsafe {
*rbuf.wrapping_offset(4 as isize) = (sector_size >> 24i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(5 as isize) = (sector_size >> 16i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(6 as isize) = (sector_size >> 8i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(7 as isize) = sector_size as u8;
}
return 8;
}
// READ CAPACITY 16 command is defined as a service action
// (SERVICE_ACTION_IN_16 command).
if ((unsafe { *scsicmd.wrapping_offset(0 as isize) }) as i32) != 158 || ((unsafe { *scsicmd.wrapping_offset(1 as isize) }) as i32) & 31 != 16 {
ata_scsi_set_invalid_field(dev, cmd, 1u16, 255u8);
return 0;
}
// sector count, 64-bit
unsafe {
*rbuf.wrapping_offset(0 as isize) = (last_lba >> 56i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(1 as isize) = (last_lba >> 48i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(2 as isize) = (last_lba >> 40i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(3 as isize) = (last_lba >> 32i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(4 as isize) = (last_lba >> 24i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(5 as isize) = (last_lba >> 16i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(6 as isize) = (last_lba >> 8i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(7 as isize) = last_lba as u8;
}
// sector size
unsafe {
*rbuf.wrapping_offset(8 as isize) = (sector_size >> 24i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(9 as isize) = (sector_size >> 16i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(10 as isize) = (sector_size >> 8i32) as u8;
}
unsafe {
*rbuf.wrapping_offset(11 as isize) = sector_size as u8;
}
if ata_dev_is_zoned(dev) {
unsafe {
*rbuf.wrapping_offset(12 as isize) = 16u8;
}
}
// RC_BASIS
unsafe {
*rbuf.wrapping_offset(13 as isize) = log2_per_phys;
}
unsafe {
*rbuf.wrapping_offset(14 as isize) = ((lowest_aligned as i32) >> 8i32 & 63) as u8;
}
unsafe {
*rbuf.wrapping_offset(15 as isize) = lowest_aligned as u8;
}
if ata_id_has_trim((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16) && (unsafe { (*dev).quirks }) & 4194304 == 0 {
let __v0: *mut u8 = rbuf.wrapping_offset(14 as isize);
unsafe {
*__v0 = ((*__v0 as i32) | 128) as u8;
}
// LBPME
if ata_id_has_zero_after_trim((unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }) as *const u16) && (unsafe { (*dev).quirks }) & 1048576 != 0 {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x016ata%u.%02u: Enabling discard_zeroes_data\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
let __v5: *mut u8 = rbuf.wrapping_offset(14 as isize);
unsafe {
*__v5 = ((*__v5 as i32) | 64) as u8;
}
// LBPRZ
}
}
16
}

/// ata_scsiop_report_luns - Simulate REPORT LUNS command
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Simulate REPORT LUNS command.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsiop_report_luns")]
extern "C" fn ata_scsiop_report_luns(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
unsafe {
*rbuf.wrapping_offset(3 as isize) = 8;
}
// just one lun, LUN 0, size 8 bytes
16
}

/// ATAPI devices typically report zero for their SCSI version, and sometimes
/// deviate from the spec WRT response data format.  If SCSI version is
/// reported as zero like normal, then we make the following fixups:
///   1) Fake MMC-5 version, to indicate to the Linux scsi midlayer this is a
/// 	modern device.
///   2) Ensure response data format / ATAPI information are always correct.
fn atapi_fixup_inquiry(cmd: *mut scsi_cmnd) {
let mut buf: [u8; 4] = [0u8; 4];
unsafe { sg_copy_to_buffer(scsi_sglist(cmd), scsi_sg_count(cmd), buf.as_mut_ptr() as *mut (), 4u64) };
if (buf[2] as i32) == 0 {
buf[2] = 5;
buf[3] = 50;
}
unsafe { sg_copy_from_buffer(scsi_sglist(cmd), scsi_sg_count(cmd), buf.as_mut_ptr() as *const (), 4u64) };
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.atapi_qc_complete")]
extern "C" fn atapi_qc_complete(qc: *mut ata_queued_cmd) {
let mut cmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
unsafe {
link = (*(*qc).dev).link;
cmd = (*qc).scsicmd;
let err_mask: u32 = (*qc).err_mask;
// handle completion from EH
if unlikely!(err_mask != 0 || (*qc).flags & 131072 != 0) != 0 {
if (*qc).flags & 131072 == 0 {
ata_gen_passthru_sense(qc);
}
// SCSI EH automatically locks door if sdev->locked is
// set.  Sometimes door lock request continues to
// fail, for example, when no media is present.  This
// creates a loop - SCSI EH issues door lock which
// fails and gets invoked again to acquire sense data
// for the failed command.
//
// If door lock fails, always clear sdev->locked to
// avoid this infinite loop.
//
// This may happen before SCSI scan is complete.  Make
// sure qc->dev->sdev isn't NULL before dereferencing.
if (*((*qc).cdb.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) == 30 && !unsafe { (*(*qc).dev).sdev }.is_null() {
(*(*(*qc).dev).sdev).__slate_bits_0.__set_locked(0u32);
}
if (*cmd).result != 0 {
ata_scsi_qc_done(qc, false, 0u32);
} else {
ata_scsi_qc_done(qc, true, 2u32);
}
} else {
if (*cmd).result != 0 {
ata_scsi_qc_done(qc, false, 0u32);
} else {
// successful completion path
if (*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as i32) == 18 && (*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) & 3 == 0 {
atapi_fixup_inquiry(cmd);
}
ata_scsi_qc_done(qc, true, 0u32);
}
}
ata_scsi_schedule_deferred_qc(link);
}
}

// atapi_xlat - Initialize PACKET taskfile
// @qc: command structure to be initialized
//
// LOCKING:
// spin_lock_irqsave(host lock)
//
// RETURNS:
// Zero on success, non-zero on failure.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.atapi_xlat")]
extern "C" fn atapi_xlat(qc: *mut ata_queued_cmd) -> u32 {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let dev: *mut ata_device = unsafe { (*qc).dev };
let nodata: i32 = (((unsafe { (*scmd).sc_data_direction }) as u32) == 3) as i32;
let mut using_pio: i32 = (nodata == 0 && (unsafe { (*dev).flags }) & 131072 != 0) as i32;
let mut nbytes: u32 = 0;
unsafe { memset(((*qc).cdb.as_mut_ptr() as *mut u8) as *mut (), 0i32, (*dev).cdb_len as u64) };
unsafe { memcpy(((*qc).cdb.as_mut_ptr() as *mut u8) as *mut (), ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const (), (*scmd).cmd_len as u64) };
unsafe {
(*qc).complete_fn = Some(atapi_qc_complete);
}
unsafe {
(*qc).tf.flags = (*qc).tf.flags | 6u64;
}
if ((unsafe { (*scmd).sc_data_direction }) as u32) == 1 {
unsafe {
(*qc).tf.flags = (*qc).tf.flags | 8;
}
}
unsafe {
(*qc).tf.__slate_anon_14.command = 160;
}
ata_qc_set_pc_nbytes(qc);
// check whether ATAPI DMA is safe
let __v6: bool = nodata == 0 && using_pio == 0 && (unsafe { atapi_check_dma(qc) }) != 0;
if __v6 {
using_pio = 1;
}
// Some controller variants snoop this value for Packet
// transfers to do state machine and FIFO management.  Thus we
// want to set it properly, and for DMA where it is
// effectively meaningless.
let mut __v7: u32;
let __UNIQUE_ID_x__708: u32 = ata_qc_raw_nbytes(qc);
let __UNIQUE_ID_y__709: u32 = 64512u32;
loop {
if __types_ok__2!(__UNIQUE_ID_x__708, __UNIQUE_ID_y__709) == 0 {
unsafe { __compiletime_assert_710() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v7 = if __UNIQUE_ID_x__708 < __UNIQUE_ID_y__709 { __UNIQUE_ID_x__708 } else { __UNIQUE_ID_y__709 };
nbytes = __v7;
// Most ATAPI devices which honor transfer chunk size don't
// behave according to the spec when odd chunk size which
// matches the transfer length is specified.  If the number of
// bytes to transfer is 2n+1.  According to the spec, what
// should happen is to indicate that 2n+1 is going to be
// transferred and transfer 2n+2 bytes where the last byte is
// padding.
//
// In practice, this doesn't happen.  ATAPI devices first
// indicate and transfer 2n bytes and then indicate and
// transfer 2 bytes where the last byte is padding.
//
// This inconsistency confuses several controllers which
// perform PIO using DMA such as Intel AHCIs and sil3124/32.
// These controllers use actual number of transferred bytes to
// update DMA pointer and transfer of 4n+2 bytes make those
// controller push DMA pointer by 4n+4 bytes because SATA data
// FISes are aligned to 4 bytes.  This causes data corruption
// and buffer overrun.
//
// Always setting nbytes to even number solves this problem
// because then ATAPI devices don't have to split data at 2n
// boundaries.
if nbytes & 1 != 0 {
nbytes = nbytes.wrapping_add(1);
}
unsafe {
(*qc).tf.lbam = (nbytes & 255) as u8;
}
unsafe {
(*qc).tf.lbah = (nbytes >> 8i32) as u8;
}
if nodata != 0 {
unsafe {
(*qc).tf.protocol = 8;
}
} else {
if using_pio != 0 {
unsafe {
(*qc).tf.protocol = 9;
}
} else {
// DMA data xfer
unsafe {
(*qc).tf.protocol = 10;
}
unsafe {
(*qc).tf.__slate_anon_8.feature = (((*qc).tf.__slate_anon_8.feature as i32) | 1) as u8;
}
if (unsafe { (*dev).flags }) & 1024 != 0 && ((unsafe { (*scmd).sc_data_direction }) as u32) != 1 {
unsafe {
(*qc).tf.__slate_anon_8.feature = (((*qc).tf.__slate_anon_8.feature as i32) | 4) as u8;
}
}
// some SATA bridges need us to indicate data xfer direction
}
}
// FIXME: We need to translate 0x05 READ_BLOCK_LIMITS to a MODE_SENSE
// as ATAPI tape drives don't get this right otherwise
0
}

fn ata_find_dev(ap: *mut ata_port, devno: u32) -> *mut ata_device {
// For the non-PMP case, ata_link_max_devices() returns 1 (SATA case),
// or 2 (IDE master + slave case). However, the former case includes
// libsas hosted devices which are numbered per scsi host, leading
// to devno potentially being larger than 0 but with each struct
// ata_device having its own struct ata_port and struct ata_link.
// To accommodate these, ignore devno and always use device number 0.
if likely!(!sata_pmp_attached(ap)) != 0 {
let link_max_devices: i32 = ata_link_max_devices((unsafe { core::ptr::addr_of_mut!((*ap).link) }) as *const ata_link);
if link_max_devices == 1 {
return unsafe { (*ap).link.device.as_mut_ptr() as *mut ata_device }.wrapping_offset(0 as isize);
}
if devno < (link_max_devices as u32) {
return unsafe { (*ap).link.device.as_mut_ptr() as *mut ata_device }.wrapping_offset(devno as isize);
}
return core::ptr::null_mut::<ata_device>();
}
// For PMP-attached devices, the device number corresponds to C
// (channel) of SCSI [H:C:I:L], indicating the port pmp link
// for the device.
if devno < ((unsafe { (*ap).nr_pmp_links }) as u32) {
return unsafe { (*unsafe { (*ap).pmp_link }.wrapping_offset(devno as isize)).device.as_mut_ptr() as *mut ata_device }.wrapping_offset(0 as isize);
}
core::ptr::null_mut::<ata_device>()
}

fn __ata_scsi_find_dev(ap: *mut ata_port, scsidev: *const scsi_device) -> *mut ata_device {
let mut devno: i32 = 0;
// skip commands not addressed to targets we simulate
if !sata_pmp_attached(ap) {
if unlikely!((unsafe { (*scsidev).channel }) != 0 || (unsafe { (*scsidev).lun }) != 0) != 0 {
return core::ptr::null_mut::<ata_device>();
}
devno = (unsafe { (*scsidev).id }) as i32;
} else {
if unlikely!((unsafe { (*scsidev).id }) != 0 || (unsafe { (*scsidev).lun }) != 0) != 0 {
return core::ptr::null_mut::<ata_device>();
}
devno = (unsafe { (*scsidev).channel }) as i32;
}
ata_find_dev(ap, devno as u32)
}

/// ata_scsi_find_dev - lookup ata_device from scsi_cmnd
/// @ap: ATA port to which the device is attached
/// @scsidev: SCSI device from which we derive the ATA device
///
/// Given various information provided in struct scsi_cmnd,
/// map that onto an ATA bus, and using that mapping
/// determine which ata_device is associated with the
/// SCSI command to be sent.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// Associated ATA device, or %NULL if not found.
#[unsafe(no_mangle)]
extern "C" fn ata_scsi_find_dev(ap: *mut ata_port, scsidev: *const scsi_device) -> *mut ata_device {
let dev: *mut ata_device = __ata_scsi_find_dev(ap, scsidev);
if !unsafe { ata_adapter_is_online(ap) } {
return core::ptr::null_mut::<ata_device>();
}
let __v0: bool = dev.is_null() || ata_dev_enabled(dev as *const ata_device) == 0;
if __v0 {
return core::ptr::null_mut::<ata_device>();
}
dev
}

/// ata_scsi_map_proto - Map pass-thru protocol value to taskfile value.
/// @byte1: Byte 1 from pass-thru CDB.
///
/// RETURNS:
/// ATA_PROT_UNKNOWN if mapping failed/unimplemented, protocol otherwise.
fn ata_scsi_map_proto(byte1: u8) -> u8 {
'__slate_break_0: {
match ((byte1 as i32) & 30) >> 1i32 {
3 => {
return 0; // Non-data
}
6 | 10 | 11 => {
return 2; // DMA
// UDMA Data-in
// UDMA Data-Out
}
4 | 5 => {
return 1; // PIO Data-in
// PIO Data-out
}
12 => {
return 6; // FPDMA
}
_ => {
break '__slate_break_0; // Hard Reset
// SRST
// Device Diagnostic
// Device Reset
// DMA Queued
// Return Response Info
// Reserved
}
}
}
255
}

/// ata_scsi_pass_thru - convert ATA pass-thru CDB to taskfile
/// @qc: command structure to be initialized
///
/// Handles either 12, 16, or 32-byte versions of the CDB.
///
/// RETURNS:
/// Zero on success, non-zero on failure.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_pass_thru")]
extern "C" fn ata_scsi_pass_thru(qc: *mut ata_queued_cmd) -> u32 {
// sanity check for pio multi commands
let mut __v2: bool = unsafe { core::mem::zeroed() };
// We may not issue NCQ commands to devices not supporting NCQ
let mut __v3: bool = unsafe { core::mem::zeroed() };
// We may not issue DMA commands if no DMA mode is set
let mut __v4: bool = unsafe { core::mem::zeroed() };
let mut cdb_offset: u16 = unsafe { core::mem::zeroed() };
let mut fp: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u16 = core::ptr::addr_of_mut!(fp) as *mut u16;
let mut cdb: *const u8 = unsafe { core::mem::zeroed() };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
let mut scmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
let mut tf: *mut ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
tf = core::ptr::addr_of_mut!((*qc).tf);
scmd = (*qc).scsicmd;
dev = (*qc).dev;
cdb = ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const u8;
let __slate_auto_bytes = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
cdb_offset = 0;
// 7Fh variable length cmd means a ata pass-thru(32)
if (*cdb.wrapping_offset(0 as isize) as i32) == 127 {
cdb_offset = 9;
}
'__join_0: {
(*tf).protocol = ata_scsi_map_proto(*cdb.wrapping_offset(1i32.wrapping_add(cdb_offset as i32) as isize));
if ((*tf).protocol as i32) == 255 {
*__slate_slot_33 = 1;
} else {
if (*cdb.wrapping_offset(2i32.wrapping_add(cdb_offset as i32) as isize) as i32) & 3 == 0 {
// When T_LENGTH is zero (No data is transferred), dir should
// be DMA_NONE.
if ((*scmd).sc_data_direction as u32) != 3 {
*__slate_slot_33 = 2i32.wrapping_add(cdb_offset as i32) as u16;
break '__join_0;
} else {
if ata_is_ncq((*tf).protocol) {
(*tf).protocol = 4;
}
}
}
// enable LBA
(*tf).flags = (*tf).flags | 16;
// 12 and 16 byte CDBs use different offsets to
// provide the various register values.
let __t1: i32 = *cdb.wrapping_offset(0 as isize) as i32;
if __t1 == 133 {
if (*cdb.wrapping_offset(1 as isize) as i32) & 1 != 0 {
(*tf).hob_feature = *cdb.wrapping_offset(3 as isize);
(*tf).hob_nsect = *cdb.wrapping_offset(5 as isize);
(*tf).hob_lbal = *cdb.wrapping_offset(7 as isize);
(*tf).hob_lbam = *cdb.wrapping_offset(9 as isize);
(*tf).hob_lbah = *cdb.wrapping_offset(11 as isize);
(*tf).flags = (*tf).flags | 1;
} else {
let __v26: u64 = (*tf).flags & 18446744073709551614u64;
(*tf).flags = __v26;
}
// 16-byte CDB - may contain extended commands.
//
// If that is the case, copy the upper byte register values.
// Always copy low byte, device and command registers.
(*tf).__slate_anon_8.feature = *cdb.wrapping_offset(4 as isize);
(*tf).nsect = *cdb.wrapping_offset(6 as isize);
(*tf).lbal = *cdb.wrapping_offset(8 as isize);
(*tf).lbam = *cdb.wrapping_offset(10 as isize);
(*tf).lbah = *cdb.wrapping_offset(12 as isize);
(*tf).device = *cdb.wrapping_offset(13 as isize);
(*tf).__slate_anon_14.command = *cdb.wrapping_offset(14 as isize);
} else {
if __t1 == 161 {
let __v20: u64 = (*tf).flags & 18446744073709551614u64;
(*tf).flags = __v20;
// 12-byte CDB - incapable of extended commands.
(*tf).__slate_anon_8.feature = *cdb.wrapping_offset(3 as isize);
(*tf).nsect = *cdb.wrapping_offset(4 as isize);
(*tf).lbal = *cdb.wrapping_offset(5 as isize);
(*tf).lbam = *cdb.wrapping_offset(6 as isize);
(*tf).lbah = *cdb.wrapping_offset(7 as isize);
(*tf).device = *cdb.wrapping_offset(8 as isize);
(*tf).__slate_anon_14.command = *cdb.wrapping_offset(9 as isize);
} else {
if (*cdb.wrapping_offset(10 as isize) as i32) & 1 != 0 {
(*tf).hob_feature = *cdb.wrapping_offset(20 as isize);
(*tf).hob_nsect = *cdb.wrapping_offset(22 as isize);
(*tf).hob_lbal = *cdb.wrapping_offset(16 as isize);
(*tf).hob_lbam = *cdb.wrapping_offset(15 as isize);
(*tf).hob_lbah = *cdb.wrapping_offset(14 as isize);
(*tf).flags = (*tf).flags | 1;
} else {
let __v17: u64 = (*tf).flags & 18446744073709551614u64;
(*tf).flags = __v17;
}
// 32-byte CDB - may contain extended command fields.
//
// If that is the case, copy the upper byte register values.
(*tf).__slate_anon_8.feature = *cdb.wrapping_offset(21 as isize);
(*tf).nsect = *cdb.wrapping_offset(23 as isize);
(*tf).lbal = *cdb.wrapping_offset(19 as isize);
(*tf).lbam = *cdb.wrapping_offset(18 as isize);
(*tf).lbah = *cdb.wrapping_offset(17 as isize);
(*tf).device = *cdb.wrapping_offset(24 as isize);
(*tf).__slate_anon_14.command = *cdb.wrapping_offset(25 as isize);
(*tf).auxiliary = get_unaligned_be32(cdb.wrapping_offset(28 as isize) as *const ());
}
}
// For NCQ commands copy the tag value
if ata_is_ncq((*tf).protocol) {
(*tf).nsect = ((*qc).hw_tag << 3i32) as u8;
}
'__join_26: {
'__join_31: {
// enforce correct master/slave bit
(*tf).device = (if (*dev).devno != 0 { ((*tf).device as i32) | 16 } else { ((*tf).device as i32) & !16 }) as u8;
let __t0: i32 = (*tf).__slate_anon_14.command as i32;
if __t0 == 34 {
break '__join_31;
} else {
if __t0 == 35 {
break '__join_31;
} else {
if __t0 == 50 {
break '__join_31;
} else {
if __t0 == 51 {
break '__join_31;
} else {
if __t0 != 56 {
if __t0 != 135 {
if __t0 != 205 {
if __t0 != 200 {
if __t0 != 37 {
if __t0 != 38 {
if __t0 != 96 {
if __t0 != 196 {
if __t0 != 41 {
if __t0 != 32 {
if __t0 != 36 {
if __t0 != 42 {
if __t0 != 43 {
if __t0 != 64 {
if __t0 != 66 {
if __t0 != 202 {
if __t0 != 53 {
if __t0 != 61 {
if __t0 != 54 {
if __t0 != 62 {
if __t0 != 97 {
if __t0 != 197 {
if __t0 != 57 {
if __t0 != 206 {
if __t0 != 48 {
if __t0 != 52 {
if __t0 != 58 {
if __t0 != 59 {
(*qc).sect_size = 512;
break '__join_26;
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
(*qc).sect_size = (*(*scmd).device).sector_size;
// XXX: case ATA_CMD_CFA_WRITE_SECTORS_WITHOUT_ERASE:
// XXX: case ATA_CMD_READ_QUEUED_EXT:
break '__join_26;
}
if ((*tf).protocol as i32) != 1 || ((*tf).nsect as i32) != 1 {
*__slate_slot_33 = 1;
break '__join_0;
} else {
(*qc).sect_size = scsi_bufflen(scmd);
}
}
// Set flags so that all registers will be written, pass on
// write indication (used for PIO/DMA setup), result TF is
// copied back and we don't whine too much about its failure.
(*tf).flags = (*tf).flags | 6u64;
if ((*scmd).sc_data_direction as u32) == 1 {
(*tf).flags = (*tf).flags | 8;
}
(*qc).flags = (*qc).flags | 80u64;
// Set transfer length.
//
// TODO: find out if we need to do more here to
//       cover scatter/gather case.
ata_qc_set_pc_nbytes(qc);
if ((*tf).protocol as i32) == 2 {
__v4 = !ata_dma_enabled(dev);
} else {
__v4 = false;
}
if __v4 {
*__slate_slot_33 = 1;
} else {
if ata_is_ncq((*tf).protocol) {
__v3 = !ata_ncq_enabled(dev);
} else {
__v3 = false;
}
if __v3 {
*__slate_slot_33 = 1;
} else {
if (*cdb.wrapping_offset(1 as isize) as i32) & 224 != 0 {
__v2 = is_multi_taskfile(tf) == 0;
} else {
__v2 = false;
}
if __v2 {
*__slate_slot_33 = 1;
} else {
if is_multi_taskfile(tf) != 0 {
let multi_count: u32 = (1i32 << ((*cdb.wrapping_offset(1 as isize) as i32) >> 5i32)) as u32;
// compare the passed through multi_count
// with the cached multi_count of libata
if multi_count != (*dev).multi_count {
while false {
}
_printk(c"\x014ata%u.%02u: invalid multi_count %u ignored\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno), multi_count);
}
}
// Filter SET_FEATURES - XFER MODE command -- otherwise,
// SET_FEATURES - XFER MODE must be preceded/succeeded
// by an update to hardware-specific registers for each
// controller (i.e. the reason for ->set_piomode(),
// ->set_dmamode(), and ->post_set_mode() hooks).
if ((*tf).__slate_anon_14.command as i32) == 239 && ((*tf).__slate_anon_8.feature as i32) == 3 {
*__slate_slot_33 = (if (*cdb.wrapping_offset(0 as isize) as i32) == 133 { 4i32 } else { 3i32 }) as u16;
} else {
// Filter TPM commands by default. These provide an
// essentially uncontrolled encrypted "back door" between
// applications and the disk. Set libata.allow_tpm=1 if you
// have a real reason for wanting to use them. This ensures
// that installed software cannot easily mess stuff up without
// user intent. DVR type users will probably ship with this enabled
// for movie content management.
//
// Note that for ATA8 we can issue a DCS change and DCS freeze lock
// for this and should do in future but that it is not sufficient as
// DCS is an optional feature set. Thus we also do the software filter
// so that we comply with the TC consortium stated goal that the user
// can turn off TC features of their system.
if ((*tf).__slate_anon_14.command as i32) >= 92 && ((*tf).__slate_anon_14.command as i32) <= 95 && libata_allow_tpm == 0 {
*__slate_slot_33 = (if (*cdb.wrapping_offset(0 as isize) as i32) == 133 { 14i32 } else { 9i32 }) as u16;
} else {
return 0;
}
}
}
}
}
}
}
ata_scsi_set_invalid_field(dev, scmd, *__slate_slot_33, 255u8);
return 1;
}
// READ/WRITE LONG use a non-standard sect_size
// commands using reported Logical Block size (e.g. 512 or 4K)
// Everything else uses 512 byte "sectors"
}

/// ata_format_dsm_trim_descr() - SATL Write Same to DSM Trim
/// @cmd: SCSI command being translated
/// @size: DSM TRIM payload size in bytes (a multiple of 512)
/// @sector: Starting sector
/// @count: Total Range of request in logical sectors
///
/// Rewrite the WRITE SAME descriptor to be a DSM TRIM little-endian formatted
/// descriptor.
///
/// The payload is a list of @size / 8 entries of the format:
///   63:48 Range Length
///   47:0  LBA
///
///  Range Length of 0 is ignored.
///  LBA's should be sorted order and not overlap.
///
/// NOTE: this is the same format as ADD LBA(S) TO NV CACHE PINNED SET
///
/// The descriptor is written straight into the WRITE SAME data-out buffer;
/// ata_dsm_trim_pages() guarantees @size does not exceed that buffer (one
/// logical block). An atomic sg_miter mapping is used so this works from the
/// command submission path and, unlike page_address(), copes with a high
/// memory payload.
///
/// Return: Number of bytes written into the data-out buffer.
fn ata_format_dsm_trim_descr(cmd: *mut scsi_cmnd, size: u64, mut sector: u64, mut count: u32) -> u64 {
let mut miter: sg_mapping_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(miter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
let mut offset: u64 = 0;
unsafe { sg_miter_start(core::ptr::addr_of_mut!(miter), scsi_sglist(cmd), scsi_sg_count(cmd), 3u32) };
loop {
let __v15: bool = offset < size && unsafe { sg_miter_next(core::ptr::addr_of_mut!(miter)) };
if !__v15 {
break;
}
let buf: *mut u64 = miter.addr as *mut u64;
let chunk: u64 = min_t__2!(u64, miter.length, size.wrapping_sub(offset));
let n: u32 = (chunk / (core::mem::size_of::<u64>() as u64)) as u32;
for i in 0..n {
let mut entry: u64 = 0;
if count != 0 {
let rlen: u32 = min_t!(u32, count, 65535i32);
entry = sector | (rlen as u64) << 48i32;
sector = sector.wrapping_add(rlen as u64);
count = count.wrapping_sub(rlen);
}
unsafe {
*buf.wrapping_offset(i as isize) = entry;
}
}
offset = offset.wrapping_add((n as u64).wrapping_mul(core::mem::size_of::<u64>() as u64));
}
unsafe { sg_miter_stop(core::ptr::addr_of_mut!(miter)) };
offset
}

// ata_scsi_write_same_xlat() - SATL Write Same to ATA SCT Write Same
// @qc: Command to be translated
//
// Translate a SCSI WRITE SAME command to be either a DSM TRIM command or
// an SCT Write Same command.
// Based on WRITE SAME has the UNMAP flag:
//
//   - When set translate to DSM TRIM
//   - When clear translate to SCT Write Same
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_write_same_xlat")]
extern "C" fn ata_scsi_write_same_xlat(qc: *mut ata_queued_cmd) -> u32 {
// For DATA SET MANAGEMENT TRIM the COUNT field (aka nsect) is the
// number of 512-byte pages to be transferred.
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_h__717: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_l__716: u32 = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_v__715: u32 = unsafe { core::mem::zeroed() };
let mut __v8: bool = unsafe { core::mem::zeroed() };
let mut unmap: u8 = unsafe { core::mem::zeroed() };
let mut bp: u8 = unsafe { core::mem::zeroed() };
let mut fp: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u16 = core::ptr::addr_of_mut!(fp) as *mut u16;
let mut n_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u32 = core::ptr::addr_of_mut!(n_block) as *mut u32;
let mut block: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut u64 = core::ptr::addr_of_mut!(block) as *mut u64;
let mut size: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u64 = core::ptr::addr_of_mut!(size) as *mut u64;
let mut n_pages: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(n_pages) as *mut u32;
let mut max_pages: u32 = unsafe { core::mem::zeroed() };
let mut cdb: *const u8 = unsafe { core::mem::zeroed() };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
let mut scmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
let mut tf: *mut ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
tf = core::ptr::addr_of_mut!((*qc).tf);
scmd = (*qc).scsicmd;
dev = (*qc).dev;
cdb = ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const u8;
max_pages = ata_dsm_trim_pages(dev);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
bp = 255;
unmap = ((*cdb.wrapping_offset(1 as isize) as i32) & 8) as u8;
// we may not issue DMA commands if no DMA mode is set
if unlikely!(!ata_dma_enabled(dev)) == 0 {
// We only allow sending this command through the block layer,
// as it modifies the DATA OUT buffer, which would corrupt user
// memory for SG_IO commands.
if unlikely!(blk_rq_is_passthrough(scsi_cmd_to_rq(scmd))) == 0 {
if unlikely!(((*scmd).cmd_len as i32) < 16) != 0 {
*__slate_slot_11 = 15;
} else {
scsi_16_lba_len(cdb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_13)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_12)));
if unmap == 0 || (*dev).quirks & 4194304 != 0 {
__v8 = true;
} else {
__v8 = !ata_id_has_trim(((*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16) as *const u16);
}
if __v8 {
*__slate_slot_11 = 1;
bp = 3;
} else {
// If the request is too large the cmd is invalid
if (*__slate_slot_12 as u64) > (max_pages.wrapping_mul(64) as u64).wrapping_mul(65535u64) {
*__slate_slot_11 = 2;
} else {
// WRITE SAME always has a sector sized buffer as payload, this
// should never be a multiple entry S/G list.
if scsi_sg_count(scmd) != 0 {
// The DATA SET MANAGEMENT TRIM payload is a whole number of 512-byte
// pages (each holding up to ATA_MAX_TRIM_RNUM LBA Range Entries),
// independent of the logical sector size. Only use as many pages as
// are needed to describe the request, capped at max_pages.
*__slate_slot_15 = ((*__slate_slot_12).wrapping_add(65535u32).wrapping_sub(1) / 65535u32).wrapping_add(64).wrapping_sub(1) / 64;
__UNIQUE_ID_v__715 = *__slate_slot_15;
__UNIQUE_ID_l__716 = 1;
__UNIQUE_ID_h__717 = max_pages;
loop {
if !(!statically_true!(__UNIQUE_ID_l__716 > __UNIQUE_ID_h__717)) {
__compiletime_assert_718();
core::hint::unreachable_unchecked();
}
if true {
break;
}
}
while false {
}
*__slate_slot_15 = if __UNIQUE_ID_v__715 >= __UNIQUE_ID_h__717 { __UNIQUE_ID_h__717 } else { if __UNIQUE_ID_v__715 <= __UNIQUE_ID_l__716 { __UNIQUE_ID_l__716 } else { __UNIQUE_ID_v__715 } };
*__slate_slot_14 = (*__slate_slot_15 as u64).wrapping_mul(512);
if ata_format_dsm_trim_descr(scmd, *__slate_slot_14, *__slate_slot_13, *__slate_slot_12) == *__slate_slot_14 {
if ata_ncq_enabled(dev) {
__v3 = ata_fpdma_dsm_supported(dev);
} else {
__v3 = false;
}
if __v3 {
// Newer devices support queued TRIM commands
(*tf).protocol = 6;
(*tf).__slate_anon_14.command = 100;
(*tf).hob_nsect = 0u8;
(*tf).nsect = ((*qc).hw_tag << 3i32) as u8;
(*tf).hob_feature = (*__slate_slot_15 >> 8i32) as u8;
(*tf).__slate_anon_8.feature = *__slate_slot_15 as u8;
(*tf).auxiliary = 1;
} else {
(*tf).protocol = 2;
(*tf).hob_feature = 0;
(*tf).__slate_anon_8.feature = 1;
(*tf).hob_nsect = (*__slate_slot_15 >> 8i32) as u8;
(*tf).nsect = *__slate_slot_15 as u8;
(*tf).__slate_anon_14.command = 6;
}
(*tf).flags = (*tf).flags | 15u64;
ata_qc_set_pc_nbytes(qc);
// The DSM TRIM payload (size) may be smaller than the WRITE SAME
// data-out buffer (one logical block); only transfer the pages that
// were actually built so the transfer length matches the COUNT field.
(*qc).nbytes = *__slate_slot_14 as u32;
return 0;
}
}
ata_scsi_set_sense(dev, scmd, 5u8, 26u8, 0u8); // "Parameter list length error"
return 1;
}
}
}
ata_scsi_set_invalid_field(dev, scmd, *__slate_slot_11, bp);
return 1;
}
}
ata_scsi_set_sense(dev, scmd, 5u8, 32u8, 0u8); // "Invalid command operation code"
return 1;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ata_scsi_cmd {
    op: u8,
    cdb_len: u8,
    sa_valid: bool,
    sa: u16,
}


/// Array of commands supported with translation or emulation, sorted in
/// ascending opcode and service action order. All of these commands are
/// processed either in ata_xlat_func() or in ata_scsi_simulate();
///
/// Note: commands that are not fully supported may be left out of this array
/// so that they are not reported as supported for passthrough but still
/// available through the block layer. For now, this includes the following
/// commands:
///  - WRITE_SAME_16: ata_scsi_write_same_xlat() forbids passthrough commands
static ata_supported_cmds: __SlateAlign16<[ata_scsi_cmd; 40]> = __SlateAlign16([{
let mut __t0: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t0.op = 0u8;
__t0.cdb_len = 6u8;
__t0
}, {
let mut __t1: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t1.op = 1u8;
__t1.cdb_len = 6u8;
__t1
}, {
let mut __t2: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t2.op = 3u8;
__t2.cdb_len = 6u8;
__t2
}, {
let mut __t3: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t3.op = 8u8;
__t3.cdb_len = 6u8;
__t3
}, {
let mut __t4: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t4.op = 10u8;
__t4.cdb_len = 6u8;
__t4
}, {
let mut __t5: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t5.op = 11u8;
__t5.cdb_len = 6u8;
__t5
}, {
let mut __t6: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t6.op = 18u8;
__t6.cdb_len = 6u8;
__t6
}, {
let mut __t7: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t7.op = 21u8;
__t7.cdb_len = 6u8;
__t7
}, {
let mut __t8: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t8.op = 26u8;
__t8.cdb_len = 6u8;
__t8
}, {
let mut __t9: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t9.op = 27u8;
__t9.cdb_len = 6u8;
__t9
}, {
let mut __t10: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t10.op = 29u8;
__t10.cdb_len = 6u8;
__t10
}, {
let mut __t11: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t11.op = 37u8;
__t11.cdb_len = 10u8;
__t11
}, {
let mut __t12: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t12.op = 40u8;
__t12.cdb_len = 10u8;
__t12
}, {
let mut __t13: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t13.op = 42u8;
__t13.cdb_len = 10u8;
__t13
}, {
let mut __t14: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t14.op = 43u8;
__t14.cdb_len = 10u8;
__t14
}, {
let mut __t15: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t15.op = 47u8;
__t15.cdb_len = 10u8;
__t15
}, {
let mut __t16: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t16.op = 53u8;
__t16.cdb_len = 10u8;
__t16
}, {
let mut __t17: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t17.op = 85u8;
__t17.cdb_len = 10u8;
__t17
}, {
let mut __t18: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t18.op = 90u8;
__t18.cdb_len = 10u8;
__t18
}, {
let mut __t19: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t19.op = 127u8;
__t19.cdb_len = 32u8;
__t19.sa_valid = 1i32 != 0i32;
__t19.sa = 8176u16;
__t19
}, {
let mut __t20: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t20.op = 133u8;
__t20.cdb_len = 16u8;
__t20
}, {
let mut __t21: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t21.op = 136u8;
__t21.cdb_len = 16u8;
__t21
}, {
let mut __t22: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t22.op = 138u8;
__t22.cdb_len = 16u8;
__t22
}, {
let mut __t23: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t23.op = 143u8;
__t23.cdb_len = 16u8;
__t23
}, {
let mut __t24: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t24.op = 145u8;
__t24.cdb_len = 16u8;
__t24
}, {
let mut __t25: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t25.op = 148u8;
__t25.cdb_len = 16u8;
__t25.sa_valid = 1i32 != 0i32;
__t25.sa = 1u16;
__t25
}, {
let mut __t26: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t26.op = 148u8;
__t26.cdb_len = 16u8;
__t26.sa_valid = 1i32 != 0i32;
__t26.sa = 2u16;
__t26
}, {
let mut __t27: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t27.op = 148u8;
__t27.cdb_len = 16u8;
__t27.sa_valid = 1i32 != 0i32;
__t27.sa = 3u16;
__t27
}, {
let mut __t28: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t28.op = 148u8;
__t28.cdb_len = 16u8;
__t28.sa_valid = 1i32 != 0i32;
__t28.sa = 4u16;
__t28
}, {
let mut __t29: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t29.op = 149u8;
__t29.cdb_len = 16u8;
__t29.sa_valid = 1i32 != 0i32;
__t29.sa = 0u16;
__t29
}, {
let mut __t30: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t30.op = 158u8;
__t30.cdb_len = 16u8;
__t30.sa_valid = 1i32 != 0i32;
__t30.sa = 16u16;
__t30
}, {
let mut __t31: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t31.op = 158u8;
__t31.cdb_len = 16u8;
__t31.sa_valid = 1i32 != 0i32;
__t31.sa = 23u16;
__t31
}, {
let mut __t32: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t32.op = 158u8;
__t32.cdb_len = 16u8;
__t32.sa_valid = 1i32 != 0i32;
__t32.sa = 24u16;
__t32
}, {
let mut __t33: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t33.op = 158u8;
__t33.cdb_len = 16u8;
__t33.sa_valid = 1i32 != 0i32;
__t33.sa = 25u16;
__t33
}, {
let mut __t34: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t34.op = 158u8;
__t34.cdb_len = 16u8;
__t34.sa_valid = 1i32 != 0i32;
__t34.sa = 26u16;
__t34
}, {
let mut __t35: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t35.op = 160u8;
__t35.cdb_len = 12u8;
__t35
}, {
let mut __t36: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t36.op = 161u8;
__t36.cdb_len = 12u8;
__t36
}, {
let mut __t37: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t37.op = 162u8;
__t37.cdb_len = 12u8;
__t37
}, {
let mut __t38: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t38.op = 163u8;
__t38.cdb_len = 12u8;
__t38.sa_valid = 1i32 != 0i32;
__t38.sa = 12u16;
__t38
}, {
let mut __t39: ata_scsi_cmd = unsafe { core::mem::zeroed() };
__t39.op = 181u8;
__t39.cdb_len = 12u8;
__t39
}]);

fn ata_scsi_get_supported_cmd(op: u8, sa: u16) -> *const ata_scsi_cmd {
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__87!() {
let cmd: *const ata_scsi_cmd = unsafe { core::ptr::addr_of!(ata_supported_cmds.0) as *const ata_scsi_cmd }.wrapping_offset(i as isize);
if ((unsafe { (*cmd).op }) as i32) == (op as i32) && ((unsafe { (*cmd).sa }) as i32) == (sa as i32) {
return cmd;
}
i = i.wrapping_add(1);
}
core::ptr::null::<ata_scsi_cmd>()
}

fn ata_scsi_supported_cmd_use_sa(op: u8) -> bool {
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__116!() {
let cmd: *const ata_scsi_cmd = unsafe { core::ptr::addr_of!(ata_supported_cmds.0) as *const ata_scsi_cmd }.wrapping_offset(i as isize);
if ((unsafe { (*cmd).op }) as i32) == (op as i32) {
return unsafe { (*cmd).sa_valid };
}
i = i.wrapping_add(1);
}
false
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ata_scsi_cmd_support {
    cdlp: u8,
    rwcdlp: u8,
}


fn ata_scsi_cmd_is_supported(dev: *mut ata_device, op: u8, sa: u16, sup: *mut ata_scsi_cmd_support) -> bool {
// First, see if we support the command.
let cmd: *const ata_scsi_cmd = ata_scsi_get_supported_cmd(op, sa);
if cmd.is_null() {
return false;
}
// Now refine the support report depending on the device features.
unsafe { memset(sup as *mut (), 0i32, core::mem::size_of::<ata_scsi_cmd_support>() as u64) };
match op as i32 {
136 => {
if (unsafe { (*dev).flags }) & 8192 != 0 {
// CDL read descriptors map to the T2A page, that is,
// rwcdlp = 0x01 and cdlp = 0x01
unsafe {
(*sup).rwcdlp = 1;
}
unsafe {
(*sup).cdlp = 1;
}
}
}
138 => {
if (unsafe { (*dev).flags }) & 8192 != 0 {
// CDL write descriptors map to the T2B page, that is,
// rwcdlp = 0x01 and cdlp = 0x02
unsafe {
(*sup).rwcdlp = 1;
}
unsafe {
(*sup).cdlp = 2;
}
}
}
148 | 149 => {
return ata_dev_is_zoned(dev);
}
158 => {
match sa as i32 {
23 | 24 => {
return (unsafe { (*dev).flags }) & 16384 != 0;
}
25 => {
return (unsafe { (*dev).flags }) & 32768 != 0;
}
26 => {
return (unsafe { (*dev).flags }) & 65536 != 0;
}
_ => {
return true;
}
}
}
162 | 181 => {
return (unsafe { (*dev).flags }) & 256 != 0;
}
_ => {
}
}
true
}

fn ata_scsi_report_all_supported_opcodes(dev: *mut ata_device, rbuf: *mut u8) -> u32 {
let mut sup: ata_scsi_cmd_support = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sup) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
}
let mut len: u32 = 4;
let mut buf: *mut u8 = rbuf.wrapping_offset(len as isize);
let mut i: i32 = 0;
while (i as u64) < ARRAY_SIZE__24!() {
if len > 2040u32 {
break;
}
let cmd: *const ata_scsi_cmd = unsafe { core::ptr::addr_of!(ata_supported_cmds.0) as *const ata_scsi_cmd }.wrapping_offset(i as isize);
// All command format
if ata_scsi_cmd_is_supported(dev, unsafe { (*cmd).op }, unsafe { (*cmd).sa }, core::ptr::addr_of_mut!(sup)) {
unsafe {
*buf.wrapping_offset(0 as isize) = (*cmd).op;
}
put_unaligned_be16(unsafe { (*cmd).sa }, buf.wrapping_offset(2 as isize) as *mut ());
unsafe {
*buf.wrapping_offset(5 as isize) = ((sup.rwcdlp as i32) << 6i32 | (sup.cdlp as i32) << 2i32) as u8;
}
if unsafe { (*cmd).sa_valid } {
let __v1: *mut u8 = buf.wrapping_offset(5 as isize);
unsafe {
*__v1 = ((*__v1 as i32) | 1) as u8;
}
}
put_unaligned_be16((unsafe { (*cmd).cdb_len }) as u16, buf.wrapping_offset(6 as isize) as *mut ());
// CTDP == 0
len = len.wrapping_add(8);
let __v6: *mut u8 = buf;
buf = __v6.wrapping_offset(8 as isize);
}
i = i.wrapping_add(1);
}
put_unaligned_be32(len.wrapping_sub(4), rbuf.wrapping_offset(0 as isize) as *mut ());
len
}

fn ata_scsi_report_supported_opcodes(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let mut sup: ata_scsi_cmd_support = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sup) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
}
let cdb: *mut u8 = unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 };
let mut sa: u16 = 0;
// All command format
// One command format with command support data, ignore sa.
// One command format, must have sa.
// One command format
match (unsafe { *cdb.wrapping_offset(2 as isize) }) as i32 {
0 => {
return ata_scsi_report_all_supported_opcodes(dev, rbuf); // All command format
}
1 => {
if ata_scsi_supported_cmd_use_sa(unsafe { *cdb.wrapping_offset(3 as isize) }) {
ata_scsi_set_invalid_field(dev, cmd, 3u16, 255u8);
return 0;
}
// One command format with command support data, ignore sa.
}
2 => {
if !ata_scsi_supported_cmd_use_sa(unsafe { *cdb.wrapping_offset(3 as isize) }) {
ata_scsi_set_invalid_field(dev, cmd, 3u16, 255u8);
return 0;
}
// One command format, must have sa.
sa = get_unaligned_be16(cdb.wrapping_offset(4 as isize) as *const ()); // One command format
}
3 => {
sa = get_unaligned_be16(cdb.wrapping_offset(4 as isize) as *const ()); // One command format
}
_ => {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014ata%u.%02u: invalid command format %d\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno), *cdb.wrapping_offset(2 as isize) as i32) };
ata_scsi_set_invalid_field(dev, cmd, 2u16, 255u8);
return 0;
}
}
// One command format
if ata_scsi_cmd_is_supported(dev, unsafe { *cdb.wrapping_offset(3 as isize) }, sa, core::ptr::addr_of_mut!(sup)) {
unsafe {
*rbuf.wrapping_offset(0 as isize) = sup.rwcdlp;
}
unsafe {
*rbuf.wrapping_offset(1 as isize) = ((sup.cdlp as i32) << 3i32 | 3) as u8;
}
} else {
unsafe {
*rbuf.wrapping_offset(1 as isize) = 1;
}
}
4
}

/// ata_scsiop_maint_in - Simulate a subset of MAINTENANCE_IN
/// @dev: Target device.
/// @cmd: SCSI command of interest.
/// @rbuf: Response buffer, to which simulated SCSI cmd output is sent.
///
/// Yields a subset to satisfy scsi_report_opcode()
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsiop_maint_in")]
extern "C" fn ata_scsiop_maint_in(dev: *mut ata_device, cmd: *mut scsi_cmnd, rbuf: *mut u8) -> u32 {
let cdb: *mut u8 = unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 };
let service_action: u8 = (((unsafe { *cdb.wrapping_offset(1 as isize) }) as i32) & 31) as u8;
match service_action as i32 {
12 => {
return ata_scsi_report_supported_opcodes(dev, cmd, rbuf);
}
_ => {
ata_scsi_set_invalid_field(dev, cmd, 1u16, 255u8);
return 0;
}
}
unsafe { core::mem::zeroed() }
}

/// ata_scsi_report_zones_complete - convert ATA output
/// @qc: command structure returning the data
///
/// Convert T-13 little-endian field representation into
/// T-10 big-endian field representation.
/// What a mess.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_report_zones_complete")]
extern "C" fn ata_scsi_report_zones_complete(qc: *mut ata_queued_cmd) {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let mut miter: sg_mapping_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(miter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
let mut bytes: u32 = 0;
lockdep_assert_held__2!(unsafe { (*(*qc).ap).lock });
unsafe { sg_miter_start(core::ptr::addr_of_mut!(miter), scsi_sglist(scmd), scsi_sg_count(scmd), 3u32) };
while unsafe { sg_miter_next(core::ptr::addr_of_mut!(miter)) } {
let mut offset: u32 = 0;
if bytes == 0 {
let mut list_length: u32 = 0;
let mut max_lba: u64 = 0;
let mut opt_lba: u64 = 0;
let mut same: u16 = 0;
// Swizzle header
let hdr__0: *mut u8 = miter.addr as *mut u8;
list_length = get_unaligned_le32(hdr__0.wrapping_offset(0 as isize) as *const ());
same = get_unaligned_le16(hdr__0.wrapping_offset(4 as isize) as *const ());
max_lba = get_unaligned_le64(hdr__0.wrapping_offset(8 as isize) as *const ());
opt_lba = get_unaligned_le64(hdr__0.wrapping_offset(16 as isize) as *const ());
put_unaligned_be32(list_length, hdr__0.wrapping_offset(0 as isize) as *mut ());
unsafe {
*hdr__0.wrapping_offset(4 as isize) = ((same as i32) & 15) as u8;
}
put_unaligned_be64(max_lba, hdr__0.wrapping_offset(8 as isize) as *mut ());
put_unaligned_be64(opt_lba, hdr__0.wrapping_offset(16 as isize) as *mut ());
offset = offset.wrapping_add(64);
bytes = bytes.wrapping_add(64);
}
while (offset as u64) < miter.length {
let mut cond: u8 = 0;
let mut r#type: u8 = 0;
let mut non_seq: u8 = 0;
let mut reset: u8 = 0;
let mut size: u64 = 0;
let mut start: u64 = 0;
let mut wp: u64 = 0;
// Swizzle zone descriptor
let rec: *mut u8 = ((unsafe { (miter.addr as *const u8).wrapping_offset(offset as isize) }) as *mut ()) as *mut u8;
r#type = (((unsafe { *rec.wrapping_offset(0 as isize) }) as i32) & 15) as u8;
cond = (((unsafe { *rec.wrapping_offset(1 as isize) }) as i32) >> 4i32 & 15) as u8;
non_seq = (((unsafe { *rec.wrapping_offset(1 as isize) }) as i32) & 2) as u8;
reset = (((unsafe { *rec.wrapping_offset(1 as isize) }) as i32) & 1) as u8;
size = get_unaligned_le64(rec.wrapping_offset(8 as isize) as *const ());
start = get_unaligned_le64(rec.wrapping_offset(16 as isize) as *const ());
wp = get_unaligned_le64(rec.wrapping_offset(24 as isize) as *const ());
unsafe {
*rec.wrapping_offset(0 as isize) = r#type;
}
unsafe {
*rec.wrapping_offset(1 as isize) = ((cond as i32) << 4i32 | (non_seq as i32) | (reset as i32)) as u8;
}
put_unaligned_be64(size, rec.wrapping_offset(8 as isize) as *mut ());
put_unaligned_be64(start, rec.wrapping_offset(16 as isize) as *mut ());
put_unaligned_be64(wp, rec.wrapping_offset(24 as isize) as *mut ());
let mut __v4: i64;
let __ret_warn_on: i32 = ((offset.wrapping_add(64) as u64) > miter.length) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("720: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 720b - ., 3; .popsection\n# {0}", const 720 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_scsi_report_zones_complete_5, sym __slate_asm_string_ata_scsi_report_zones_complete_6, const 4119 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("721: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 721b - ., 4; .popsection\n# {0}", const 721 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
offset = offset.wrapping_add(64);
bytes = bytes.wrapping_add(64);
}
}
unsafe { sg_miter_stop(core::ptr::addr_of_mut!(miter)) };
ata_scsi_qc_complete(qc);
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_zbc_in_xlat")]
extern "C" fn ata_scsi_zbc_in_xlat(qc: *mut ata_queued_cmd) -> u32 {
let mut __v6: bool = unsafe { core::mem::zeroed() };
let mut n_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(n_block) as *mut u32;
let mut block: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut u64 = core::ptr::addr_of_mut!(block) as *mut u64;
let mut bp: u8 = unsafe { core::mem::zeroed() };
let mut options: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut u8 = core::ptr::addr_of_mut!(options) as *mut u8;
let mut sa: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u8 = core::ptr::addr_of_mut!(sa) as *mut u8;
let mut fp: u16 = unsafe { core::mem::zeroed() };
let mut sect: core::mem::MaybeUninit<u16> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut u16 = core::ptr::addr_of_mut!(sect) as *mut u16;
let mut cdb: *const u8 = unsafe { core::mem::zeroed() };
let mut scmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
let mut tf: *mut ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
'__join_1: {
tf = core::ptr::addr_of_mut!((*qc).tf);
scmd = (*qc).scsicmd;
cdb = ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const u8;
let __slate_auto_bytes = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 2usize);
fp = 1i32.wrapping_neg() as u16;
let __slate_auto_bytes_1 = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_14 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
bp = 255;
let __slate_auto_bytes_3 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if unlikely!(((*scmd).cmd_len as i32) < 16) != 0 {
while false {
}
_printk(c"\x014ata%u.%02u: invalid cdb length %d\n".as_ptr() as *const u8, (*(*(*(*qc).dev).link).ap).print_id, ((*(*(*qc).dev).link).pmp as u32).wrapping_add((*(*qc).dev).devno), (*scmd).cmd_len as i32);
fp = 15;
} else {
scsi_16_lba_len(cdb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_12)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_11)));
if *__slate_slot_11 != scsi_bufflen(scmd) {
while false {
}
_printk(c"\x014ata%u.%02u: non-matching transfer count (%d/%d)\n".as_ptr() as *const u8, (*(*(*(*qc).dev).link).ap).print_id, ((*(*(*qc).dev).link).pmp as u32).wrapping_add((*(*qc).dev).devno), *__slate_slot_11, scsi_bufflen(scmd));
} else {
*__slate_slot_15 = ((*cdb.wrapping_offset(1 as isize) as i32) & 31) as u8;
if (*__slate_slot_15 as i32) != 0 {
while false {
}
_printk(c"\x014ata%u.%02u: invalid service action %d\n".as_ptr() as *const u8, (*(*(*(*qc).dev).link).ap).print_id, ((*(*(*qc).dev).link).pmp as u32).wrapping_add((*(*qc).dev).devno), *__slate_slot_15 as i32);
fp = 1;
break '__join_1;
} else {
// ZAC allows only for transfers in 512 byte blocks,
// and uses a 16 bit value for the transfer count.
if *__slate_slot_11 / 512 > 65535 || *__slate_slot_11 < 512 || *__slate_slot_11 % 512 != 0 {
while false {
}
_printk(c"\x014ata%u.%02u: invalid transfer count %d\n".as_ptr() as *const u8, (*(*(*(*qc).dev).link).ap).print_id, ((*(*(*qc).dev).link).pmp as u32).wrapping_add((*(*qc).dev).devno), *__slate_slot_11);
} else {
*__slate_slot_17 = (*__slate_slot_11 / 512) as u16;
*__slate_slot_14 = ((*cdb.wrapping_offset(14 as isize) as i32) & 191) as u8;
if ata_ncq_enabled((*qc).dev) {
__v6 = ata_fpdma_zac_mgmt_in_supported((*qc).dev);
} else {
__v6 = false;
}
if __v6 {
(*tf).protocol = 6;
(*tf).__slate_anon_14.command = 101;
(*tf).hob_nsect = 2u8;
(*tf).nsect = ((*qc).hw_tag << 3i32) as u8;
(*tf).__slate_anon_8.feature = ((*__slate_slot_17 as i32) & 255) as u8;
(*tf).hob_feature = ((*__slate_slot_17 as i32) >> 8i32 & 255) as u8;
(*tf).auxiliary = (0 | (*__slate_slot_14 as i32) << 8i32) as u32;
} else {
(*tf).__slate_anon_14.command = 74;
(*tf).__slate_anon_8.feature = 0;
(*tf).protocol = 2;
(*tf).hob_feature = *__slate_slot_14;
(*tf).hob_nsect = ((*__slate_slot_17 as i32) >> 8i32 & 255) as u8;
(*tf).nsect = ((*__slate_slot_17 as i32) & 255) as u8;
}
(*tf).device = 64;
(*tf).lbah = (*__slate_slot_12 >> 16i32 & 255) as u8;
(*tf).lbam = (*__slate_slot_12 >> 8i32 & 255) as u8;
(*tf).lbal = (*__slate_slot_12 & 255) as u8;
(*tf).hob_lbah = (*__slate_slot_12 >> 40i32 & 255) as u8;
(*tf).hob_lbam = (*__slate_slot_12 >> 32i32 & 255) as u8;
(*tf).hob_lbal = (*__slate_slot_12 >> 24i32 & 255) as u8;
(*tf).flags = (*tf).flags | 7u64;
(*qc).flags = (*qc).flags | 16;
ata_qc_set_pc_nbytes(qc);
(*qc).complete_fn = Some(ata_scsi_report_zones_complete);
return 0;
}
}
}
ata_scsi_set_sense((*qc).dev, scmd, 5u8, 26u8, 0u8); // "Parameter list length error"
return 1;
}
}
ata_scsi_set_invalid_field((*qc).dev, scmd, fp, bp);
return 1;
}
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_zbc_out_xlat")]
extern "C" fn ata_scsi_zbc_out_xlat(qc: *mut ata_queued_cmd) -> u32 {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut fp: u16 = unsafe { core::mem::zeroed() };
let mut n_block: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut u32 = core::ptr::addr_of_mut!(n_block) as *mut u32;
let mut block: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(block) as *mut u64;
let mut sa: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u8 = core::ptr::addr_of_mut!(sa) as *mut u8;
let mut all: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u8 = core::ptr::addr_of_mut!(all) as *mut u8;
let mut cdb: *const u8 = unsafe { core::mem::zeroed() };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
let mut scmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
let mut tf: *mut ata_taskfile = unsafe { core::mem::zeroed() };
unsafe {
'__join_0: {
tf = core::ptr::addr_of_mut!((*qc).tf);
scmd = (*qc).scsicmd;
dev = (*qc).dev;
cdb = ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const u8;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
fp = 1i32.wrapping_neg() as u16;
if unlikely!(((*scmd).cmd_len as i32) < 16) != 0 {
fp = 15;
} else {
*__slate_slot_7 = ((*cdb.wrapping_offset(1 as isize) as i32) & 31) as u8;
if (*__slate_slot_7 as i32) != 1 && (*__slate_slot_7 as i32) != 2 && (*__slate_slot_7 as i32) != 3 && (*__slate_slot_7 as i32) != 4 {
fp = 1;
} else {
scsi_16_lba_len(cdb, &mut (*core::ptr::addr_of_mut!(*__slate_slot_6)), &mut (*core::ptr::addr_of_mut!(*__slate_slot_5)));
if *__slate_slot_5 != 0 {
// ZAC MANAGEMENT OUT doesn't define any length
ata_scsi_set_sense((*qc).dev, scmd, 5u8, 26u8, 0u8); // "Parameter list length error"
return 1;
} else {
*__slate_slot_8 = ((*cdb.wrapping_offset(14 as isize) as i32) & 1) as u8;
if *__slate_slot_8 != 0 {
// Ignore the block address (zone ID) as defined by ZBC.
*__slate_slot_6 = 0;
} else {
if *__slate_slot_6 >= (*dev).n_sectors {
// Block must be a valid zone ID (a zone start LBA).
fp = 2;
break '__join_0;
}
}
if ata_ncq_enabled((*qc).dev) {
__v3 = ata_fpdma_zac_mgmt_out_supported((*qc).dev);
} else {
__v3 = false;
}
if __v3 {
(*tf).protocol = 4;
(*tf).__slate_anon_14.command = 99;
(*tf).__slate_anon_8.feature = 7;
(*tf).nsect = ((*qc).hw_tag << 3i32) as u8;
(*tf).auxiliary = ((*__slate_slot_7 as i32) | ((*__slate_slot_8 as u16) as i32) << 8i32) as u32;
} else {
(*tf).protocol = 0;
(*tf).__slate_anon_14.command = 159;
(*tf).__slate_anon_8.feature = *__slate_slot_7;
(*tf).hob_feature = *__slate_slot_8;
}
(*tf).lbah = (*__slate_slot_6 >> 16i32 & 255) as u8;
(*tf).lbam = (*__slate_slot_6 >> 8i32 & 255) as u8;
(*tf).lbal = (*__slate_slot_6 & 255) as u8;
(*tf).hob_lbah = (*__slate_slot_6 >> 40i32 & 255) as u8;
(*tf).hob_lbam = (*__slate_slot_6 >> 32i32 & 255) as u8;
(*tf).hob_lbal = (*__slate_slot_6 >> 24i32 & 255) as u8;
(*tf).device = 64;
(*tf).flags = (*tf).flags | 7u64;
return 0;
}
}
}
}
ata_scsi_set_invalid_field((*qc).dev, scmd, fp, 255u8);
return 1;
}
}

// ata_mselect_caching - Simulate MODE SELECT for caching info page
// @qc: Storage for translated ATA taskfile
// @buf: input buffer
// @len: number of valid bytes in the input buffer
// @fp: out parameter for the failed field on error
//
// Prepare a taskfile to modify caching information for the device.
//
// LOCKING:
// None.
fn ata_mselect_caching(qc: *mut ata_queued_cmd, buf: *const u8, len: i32, fp: &mut u16) -> i32 {
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).tf) };
let dev: *mut ata_device = unsafe { (*qc).dev };
let mut mpage: __SlateAlign16<[u8; 20]> = __SlateAlign16([0u8; 20]);
let mut wce: u8 = 0;
let mut i: i32 = 0;
// The first two bytes of def_cache_mpage are a header, so offsets
// in mpage are off by 2 compared to buf.  Same for len.
if len != 18i32 {
let mut __v0: i32;
let __UNIQUE_ID_x__722: i32 = len;
let __UNIQUE_ID_y__723: i32 = 18i32;
loop {
if __types_ok!(__UNIQUE_ID_x__722, __UNIQUE_ID_y__723) == 0 {
unsafe { __compiletime_assert_724() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__722 < __UNIQUE_ID_y__723 { __UNIQUE_ID_x__722 } else { __UNIQUE_ID_y__723 };
unsafe {
*(fp as *mut u16) = __v0 as u16;
}
return 22i32.wrapping_neg();
}
wce = (((unsafe { *buf.wrapping_offset(0 as isize) }) as i32) & 4i32) as u8;
// Check that read-only bits are not modified.
ata_msense_caching(unsafe { (*dev).__slate_anon_26.id.as_mut_ptr() as *mut u16 }, mpage.0.as_mut_ptr() as *mut u8, false);
while i < 18i32 {
if i != 0 {
if (mpage.0[i.wrapping_add(2) as usize] as i32) != ((unsafe { *buf.wrapping_offset(i as isize) }) as i32) {
unsafe {
*(fp as *mut u16) = i as u16;
}
return 22i32.wrapping_neg();
}
}
i = i.wrapping_add(1);
}
unsafe {
(*tf).flags = (*tf).flags | 6u64;
}
unsafe {
(*tf).protocol = 0;
}
unsafe {
(*tf).nsect = 0;
}
unsafe {
(*tf).__slate_anon_14.command = 239;
}
unsafe {
(*tf).__slate_anon_8.feature = (if wce != 0 { 2i32 } else { 130i32 }) as u8;
}
0
}

// Simulate MODE SELECT control mode page, sub-page 0.
fn ata_mselect_control_spg0(qc: *mut ata_queued_cmd, buf: *const u8, len: i32, fp: &mut u16) -> i32 {
let dev: *mut ata_device = unsafe { (*qc).dev };
let mut mpage: [u8; 12] = [0u8; 12];
let mut d_sense: u8 = 0;
let mut i: i32 = 0;
// The first two bytes of def_control_mpage are a header, so offsets
// in mpage are off by 2 compared to buf.  Same for len.
if len != 10i32 {
let mut __v0: i32;
let __UNIQUE_ID_x__725: i32 = len;
let __UNIQUE_ID_y__726: i32 = 10i32;
loop {
if __types_ok!(__UNIQUE_ID_x__725, __UNIQUE_ID_y__726) == 0 {
unsafe { __compiletime_assert_727() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__725 < __UNIQUE_ID_y__726 { __UNIQUE_ID_x__725 } else { __UNIQUE_ID_y__726 };
unsafe {
*(fp as *mut u16) = __v0 as u16;
}
return 22i32.wrapping_neg();
}
d_sense = (((unsafe { *buf.wrapping_offset(0 as isize) }) as i32) & 4i32) as u8;
// Check that read-only bits are not modified.
ata_msense_control_spg0(dev, mpage.as_mut_ptr() as *mut u8, false);
while i < 10i32 {
if i != 0 {
if (mpage[2i32.wrapping_add(i) as usize] as i32) != ((unsafe { *buf.wrapping_offset(i as isize) }) as i32) {
unsafe {
*(fp as *mut u16) = i as u16;
}
return 22i32.wrapping_neg();
}
}
i = i.wrapping_add(1);
}
if (d_sense as i32) & 4i32 != 0 {
unsafe {
(*dev).flags = (*dev).flags | 2147483648;
}
} else {
let __v10: u64 = (unsafe { (*dev).flags }) & 2147483647u64;
unsafe {
(*dev).flags = __v10;
}
}
0
}

// Translate MODE SELECT control mode page, sub-page f2h (ATA feature mode
// page) into a SET FEATURES command.
fn ata_mselect_control_ata_feature(qc: *mut ata_queued_cmd, buf: *const u8, len: i32, fp: &mut u16) -> i32 {
let dev: *mut ata_device = unsafe { (*qc).dev };
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).tf) };
let mut cdl_action: u8 = 0;
// The first four bytes of ATA Feature Control mode page are a header,
// so offsets in mpage are off by 4 compared to buf.  Same for len.
if len != 12i32 {
let mut __v0: i32;
let __UNIQUE_ID_x__728: i32 = len;
let __UNIQUE_ID_y__729: i32 = 12i32;
loop {
if __types_ok!(__UNIQUE_ID_x__728, __UNIQUE_ID_y__729) == 0 {
unsafe { __compiletime_assert_730() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__728 < __UNIQUE_ID_y__729 { __UNIQUE_ID_x__728 } else { __UNIQUE_ID_y__729 };
unsafe {
*(fp as *mut u16) = __v0 as u16;
}
return 22i32.wrapping_neg();
}
// Check cdl_ctrl
match ((unsafe { *buf.wrapping_offset(0 as isize) }) as i32) & 3 {
0 => {
let mut __v2: i32;
if false {
unsafe { _printk(c"\x017ata%u.%02u: Disabling CDL\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
}
__v2 = 0;
// Disable CDL
cdl_action = 0;
let __v5: u64 = (unsafe { (*dev).flags }) & 4278190079u64;
unsafe {
(*dev).flags = __v5;
}
}
2 => {
if (unsafe { (*dev).flags }) & 8388608 != 0 {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x013ata%u.%02u: NCQ priority must be disabled to enable CDL\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
return 22i32.wrapping_neg();
}
// Enable CDL. Since CDL is mutually exclusive with NCQ
// priority, allow this only if NCQ priority is disabled.
let mut __v9: i32;
if false {
unsafe { _printk(c"\x017ata%u.%02u: Enabling CDL\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno)) };
}
__v9 = 0;
cdl_action = 1;
unsafe {
(*dev).flags = (*dev).flags | 16777216;
}
}
_ => {
unsafe {
*(fp as *mut u16) = 0;
}
return 22i32.wrapping_neg();
}
}
unsafe {
(*tf).flags = (*tf).flags | 6u64;
}
unsafe {
(*tf).protocol = 0;
}
unsafe {
(*tf).__slate_anon_14.command = 239;
}
unsafe {
(*tf).__slate_anon_8.feature = 13;
}
unsafe {
(*tf).nsect = cdl_action;
}
1
}

/// ata_mselect_control - Simulate MODE SELECT for control page
/// @qc: Storage for translated ATA taskfile
/// @spg: target sub-page of the control page
/// @buf: input buffer
/// @len: number of valid bytes in the input buffer
/// @fp: out parameter for the failed field on error
///
/// Prepare a taskfile to modify caching information for the device.
///
/// LOCKING:
/// None.
fn ata_mselect_control(qc: *mut ata_queued_cmd, spg: u8, buf: *const u8, len: i32, fp: &mut u16) -> i32 {
match spg as i32 {
0 => {
return ata_mselect_control_spg0(qc, buf, len, fp);
}
242 => {
return ata_mselect_control_ata_feature(qc, buf, len, fp);
}
_ => {
return 22i32.wrapping_neg();
}
}
unsafe { core::mem::zeroed() }
}

/// ata_scsi_mode_select_xlat - Simulate MODE SELECT 6, 10 commands
/// @qc: Storage for translated ATA taskfile
///
/// Converts a MODE SELECT command to an ATA SET FEATURES taskfile.
/// Assume this is invoked for direct access devices (e.g. disks) only.
/// There should be no block descriptor for other device types.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_mode_select_xlat")]
extern "C" fn ata_scsi_mode_select_xlat(qc: *mut ata_queued_cmd) -> u32 {
// Test early for possible overrun.
let mut __v26: bool = unsafe { core::mem::zeroed() };
let mut p: *const u8 = unsafe { core::mem::zeroed() };
let mut buffer: core::mem::MaybeUninit<__SlateAlign16<[u8; 64]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_28: *mut [u8; 64] = core::ptr::addr_of_mut!(buffer) as *mut [u8; 64];
let mut bp: u8 = unsafe { core::mem::zeroed() };
let mut fp: u16 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_31: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut len: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_32: *mut i32 = core::ptr::addr_of_mut!(len) as *mut i32;
let mut bd_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_33: *mut u32 = core::ptr::addr_of_mut!(bd_len) as *mut u32;
let mut hdr_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_34: *mut u32 = core::ptr::addr_of_mut!(hdr_len) as *mut u32;
let mut pg_len: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_35: *mut u32 = core::ptr::addr_of_mut!(pg_len) as *mut u32;
let mut six_byte: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_36: *mut u32 = core::ptr::addr_of_mut!(six_byte) as *mut u32;
let mut spg: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_37: *mut u8 = core::ptr::addr_of_mut!(spg) as *mut u8;
let mut pg: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_38: *mut u8 = core::ptr::addr_of_mut!(pg) as *mut u8;
let mut cdb: *const u8 = unsafe { core::mem::zeroed() };
let mut scmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
unsafe {
'__join_3: {
scmd = (*qc).scsicmd;
cdb = ((*scmd).cmnd.as_mut_ptr() as *mut u8) as *const u8;
let __slate_auto_bytes = __slate_slot_38 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_1 = __slate_slot_37 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 1usize);
let __slate_auto_bytes_2 = __slate_slot_36 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_35 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_4 = __slate_slot_34 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_33 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_32 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_31 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 4usize);
fp = 1i32.wrapping_neg() as u16;
bp = 255;
let __slate_auto_bytes = __slate_slot_28 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 64usize);
p = (*__slate_slot_28).as_mut_ptr() as *const u8;
*__slate_slot_36 = ((*cdb.wrapping_offset(0 as isize) as i32) == 21) as u32;
if *__slate_slot_36 != 0 {
if ((*scmd).cmd_len as i32) < 5 {
fp = 4;
break '__join_3;
} else {
*__slate_slot_32 = *cdb.wrapping_offset(4 as isize) as i32;
*__slate_slot_34 = 4;
}
} else {
if ((*scmd).cmd_len as i32) < 9 {
fp = 8;
break '__join_3;
} else {
*__slate_slot_32 = get_unaligned_be16(cdb.wrapping_offset(7 as isize) as *const ()) as i32;
*__slate_slot_34 = 8;
}
}
// We only support PF=1, SP=0.
if (*cdb.wrapping_offset(1 as isize) as i32) & 17 != 16 {
fp = 1;
bp = (if (*cdb.wrapping_offset(1 as isize) as i32) & 1 != 0 { 1i32 } else { 5i32 }) as u8;
} else {
if scsi_sg_count(scmd) == 0 {
__v26 = true;
} else {
__v26 = (*scsi_sglist(scmd)).length < (*__slate_slot_32 as u32);
}
'__join_1: {
if !__v26 {
// Move past header and block descriptors.
if (*__slate_slot_32 as u32) >= *__slate_slot_34 {
if sg_copy_to_buffer(scsi_sglist(scmd), scsi_sg_count(scmd), (*__slate_slot_28).as_mut_ptr() as *mut (), core::mem::size_of::<[u8; 64]>() as u64) != 0 {
if *__slate_slot_36 != 0 {
*__slate_slot_33 = *p.wrapping_offset(3 as isize) as u32;
} else {
*__slate_slot_33 = get_unaligned_be16(p.wrapping_offset(6 as isize) as *const ()) as u32;
}
*__slate_slot_32 = (*__slate_slot_32 as u32).wrapping_sub(*__slate_slot_34) as i32;
let __v23: *const u8 = p;
p = __v23.wrapping_offset(*__slate_slot_34 as isize);
if (*__slate_slot_32 as u32) >= *__slate_slot_33 {
'__join_2: {
if *__slate_slot_33 != 0 && *__slate_slot_33 != 8 {
fp = (if *__slate_slot_36 != 0 { 3i32 } else { 6i32 }) as u16;
fp = ((fp as i32) as u32).wrapping_add((*__slate_slot_33).wrapping_add(*__slate_slot_34)) as u16;
} else {
'__join_0: {
*__slate_slot_32 = (*__slate_slot_32 as u32).wrapping_sub(*__slate_slot_33) as i32;
let __v17: *const u8 = p;
p = __v17.wrapping_offset(*__slate_slot_33 as isize);
if *__slate_slot_32 != 0 {
// Parse both possible formats for the mode page headers.
*__slate_slot_38 = ((*p.wrapping_offset(0 as isize) as i32) & 63) as u8;
if (*p.wrapping_offset(0 as isize) as i32) & 64 != 0 {
if *__slate_slot_32 < 4 {
break '__join_1;
} else {
*__slate_slot_37 = *p.wrapping_offset(1 as isize);
*__slate_slot_35 = get_unaligned_be16(p.wrapping_offset(2 as isize) as *const ()) as u32;
let __v11: *const u8 = p;
p = __v11.wrapping_offset(4 as isize);
let __v9: i32 = *__slate_slot_32;
*__slate_slot_32 = __v9.wrapping_sub(4);
}
} else {
if *__slate_slot_32 < 2 {
break '__join_1;
} else {
*__slate_slot_37 = 0;
*__slate_slot_35 = *p.wrapping_offset(1 as isize) as u32;
let __v15: *const u8 = p;
p = __v15.wrapping_offset(2 as isize);
let __v13: i32 = *__slate_slot_32;
*__slate_slot_32 = __v13.wrapping_sub(2);
}
}
'__join_14: {
// Supported subpages: all subpages and ATA feature sub-page f2h of
// the control page.
if *__slate_slot_37 != 0 {
let __t1: i32 = *__slate_slot_37 as i32;
if __t1 == 255 {
if (*__slate_slot_38 as i32) == 10 {
fp = (if (*p.wrapping_offset(0 as isize) as i32) & 64 != 0 { 1i32 } else { 0i32 }) as u16;
fp = ((fp as i32) as u32).wrapping_add((*__slate_slot_34).wrapping_add(*__slate_slot_33)) as u16;
break '__join_2;
} else {
// All subpages is not supported for the control page
break '__join_14;
}
} else {
if __t1 == 242 {
if (*(*qc).dev).flags & 8192 != 0 && (*__slate_slot_38 as i32) == 10 {
break '__join_14;
}
}
}
fp = (if (*p.wrapping_offset(0 as isize) as i32) & 64 != 0 { 1i32 } else { 0i32 }) as u16;
fp = ((fp as i32) as u32).wrapping_add((*__slate_slot_34).wrapping_add(*__slate_slot_33)) as u16;
break '__join_2;
}
}
if *__slate_slot_35 > (*__slate_slot_32 as u32) {
break '__join_1;
} else {
let __t0: i32 = *__slate_slot_38 as i32;
if __t0 == 8 {
if ata_mselect_caching(qc, p, *__slate_slot_35 as i32, &mut (*core::ptr::addr_of_mut!(fp))) < 0 {
fp = ((fp as i32) as u32).wrapping_add((*__slate_slot_34).wrapping_add(*__slate_slot_33)) as u16;
break '__join_2;
}
} else {
if __t0 == 10 {
*__slate_slot_31 = ata_mselect_control(qc, *__slate_slot_37, p, *__slate_slot_35 as i32, &mut (*core::ptr::addr_of_mut!(fp)));
if *__slate_slot_31 < 0 {
fp = ((fp as i32) as u32).wrapping_add((*__slate_slot_34).wrapping_add(*__slate_slot_33)) as u16;
break '__join_2;
} else {
if *__slate_slot_31 == 0 {
break '__join_0;
}
// No ATA command to send
}
} else {
fp = (*__slate_slot_33).wrapping_add(*__slate_slot_34) as u16; // Invalid page code
break '__join_2;
}
}
// Only one page has changeable data, so we only support setting one
// page at a time.
if (*__slate_slot_32 as u32) > *__slate_slot_35 {
break '__join_2;
} else {
return 0;
}
}
}
}
(*scmd).result = 0;
return 1;
}
}
ata_scsi_set_invalid_parameter((*qc).dev, scmd, fp);
return 1;
}
}
}
}
}
ata_scsi_set_sense((*qc).dev, scmd, 5u8, 26u8, 0u8); // "Parameter list length error"
return 1;
}
}
ata_scsi_set_invalid_field((*qc).dev, scmd, fp, bp);
return 1;
}
}

fn ata_scsi_trusted_op(len: u32, send: bool, dma: bool) -> u8 {
if len == 0 {
return 91;
} else {
if send {
return (if dma { 95i32 } else { 94i32 }) as u8;
} else {
return (if dma { 93i32 } else { 92i32 }) as u8;
}
}
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_security_inout_xlat")]
extern "C" fn ata_scsi_security_inout_xlat(qc: *mut ata_queued_cmd) -> u32 {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let cdb: *const u8 = (unsafe { (*scmd).cmnd.as_mut_ptr() as *mut u8 }) as *const u8;
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).tf) };
let secp: u8 = unsafe { *cdb.wrapping_offset(1 as isize) };
let send: bool = ((unsafe { *cdb.wrapping_offset(0 as isize) }) as i32) == 181;
let spsp: u16 = get_unaligned_be16(cdb.wrapping_offset(2 as isize) as *const ());
let mut len: u32 = get_unaligned_be32(cdb.wrapping_offset(6 as isize) as *const ());
let dma: bool = (unsafe { (*(*qc).dev).flags }) & 131072 == 0;
// We don't support the ATA "security" protocol.
if (secp as i32) == 239 {
ata_scsi_set_invalid_field(unsafe { (*qc).dev }, scmd, 1u16, 0u8);
return 1;
}
if ((unsafe { *cdb.wrapping_offset(4 as isize) }) as i32) & 7 != 0 {
// INC_512
if len > 65535 {
ata_scsi_set_invalid_field(unsafe { (*qc).dev }, scmd, 6u16, 0u8);
return 1;
}
} else {
if len > 33553920 {
ata_scsi_set_invalid_field(unsafe { (*qc).dev }, scmd, 6u16, 0u8);
return 1;
}
// convert to the sector-based ATA addressing
len = len.wrapping_add(511) / 512;
}
unsafe {
(*tf).protocol = (if dma { 2i32 } else { 1i32 }) as u8;
}
unsafe {
(*tf).flags = (*tf).flags | 22u64;
}
if send {
unsafe {
(*tf).flags = (*tf).flags | 8;
}
}
unsafe {
(*tf).__slate_anon_14.command = ata_scsi_trusted_op(len, send, dma);
}
unsafe {
(*tf).__slate_anon_8.feature = secp;
}
unsafe {
(*tf).lbam = ((spsp as i32) & 255) as u8;
}
unsafe {
(*tf).lbah = ((spsp as i32) >> 8i32) as u8;
}
if len != 0 {
unsafe {
(*tf).nsect = (len & 255) as u8;
}
unsafe {
(*tf).lbal = (len >> 8i32) as u8;
}
} else {
if !send {
unsafe {
(*tf).lbah = 128u8;
}
}
}
ata_qc_set_pc_nbytes(qc);
0
}

/// Convert T-13 little-endian field representation of GET PHYSICAL ELEMENT
/// STATUS DMA command reply into T-10 big-endian field representation.
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_get_phys_element_status_complete")]
extern "C" fn ata_scsi_get_phys_element_status_complete(qc: *mut ata_queued_cmd) {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let mut miter: sg_mapping_iter = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(miter) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
let mut bytes: u32 = 0;
lockdep_assert_held__2!(unsafe { (*(*qc).ap).lock });
unsafe { sg_miter_start(core::ptr::addr_of_mut!(miter), scsi_sglist(scmd), scsi_sg_count(scmd), 3u32) };
while unsafe { sg_miter_next(core::ptr::addr_of_mut!(miter)) } {
let mut offset: u32 = 0;
if bytes == 0 {
let mut num_desc: u32 = 0;
let mut num_desc_returned: u32 = 0;
let mut id: u32 = 0;
let mut max_depop: u16 = 0;
let mut cur_depop: u16 = 0;
// Swizzle the header
let hdr__0: *mut u8 = miter.addr as *mut u8;
num_desc = get_unaligned_le32(hdr__0.wrapping_offset(0 as isize) as *const ());
num_desc_returned = get_unaligned_le32(hdr__0.wrapping_offset(4 as isize) as *const ());
id = get_unaligned_le32(hdr__0.wrapping_offset(8 as isize) as *const ());
max_depop = get_unaligned_le16(hdr__0.wrapping_offset(12 as isize) as *const ());
cur_depop = get_unaligned_le16(hdr__0.wrapping_offset(14 as isize) as *const ());
put_unaligned_be32(num_desc, hdr__0.wrapping_offset(0 as isize) as *mut ());
put_unaligned_be32(num_desc_returned, hdr__0.wrapping_offset(4 as isize) as *mut ());
put_unaligned_be32(id, hdr__0.wrapping_offset(8 as isize) as *mut ());
put_unaligned_be16(max_depop, hdr__0.wrapping_offset(12 as isize) as *mut ());
put_unaligned_be16(cur_depop, hdr__0.wrapping_offset(14 as isize) as *mut ());
offset = offset.wrapping_add(32);
bytes = bytes.wrapping_add(32);
}
// Swizzle the descriptors.
while (offset as u64) < miter.length {
let mut id: u32 = 0;
let mut r#type: u8 = 0;
let desc: *mut u8 = ((unsafe { (miter.addr as *const u8).wrapping_offset(offset as isize) }) as *mut ()) as *mut u8;
id = get_unaligned_le32(desc.wrapping_offset(4 as isize) as *const ());
put_unaligned_be32(id, desc.wrapping_offset(4 as isize) as *mut ());
r#type = unsafe { *desc.wrapping_offset(14 as isize) };
if (r#type as i32) == 1 {
let capacity: u64 = get_unaligned_le64(desc.wrapping_offset(16 as isize) as *const ());
put_unaligned_be64(capacity, desc.wrapping_offset(16 as isize) as *mut ());
} else {
let mut num_zones: u64 = 0;
id = get_unaligned_le32(desc.wrapping_offset(16 as isize) as *const ());
num_zones = get_unaligned_le64(desc.wrapping_offset(24 as isize) as *const ());
put_unaligned_be32(id, desc.wrapping_offset(16 as isize) as *mut ());
put_unaligned_be64(num_zones, desc.wrapping_offset(24 as isize) as *mut ());
}
offset = offset.wrapping_add(32);
bytes = bytes.wrapping_add(32);
}
}
unsafe { sg_miter_stop(core::ptr::addr_of_mut!(miter)) };
ata_scsi_qc_complete(qc);
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_get_phys_element_status_xlat")]
extern "C" fn ata_scsi_get_phys_element_status_xlat(qc: *mut ata_queued_cmd) -> u32 {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let cdb: *const u8 = (unsafe { (*scmd).cmnd.as_mut_ptr() as *mut u8 }) as *const u8;
let dev: *mut ata_device = unsafe { (*qc).dev };
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).tf) };
let mut starting_element: u32 = 0;
let mut len: u32 = 0;
// ATA_CMD_GET_PHYS_ELEMENT_STATUS is a DMA command.
let __v0: bool = (unsafe { (*dev).flags }) & 16384 == 0 || !ata_dma_enabled(dev);
if __v0 {
ata_scsi_set_sense(dev, scmd, 5u8, 32u8, 0u8);
return 1;
}
len = get_unaligned_be32(cdb.wrapping_offset(10 as isize) as *const ()) / 512;
if len == 0 || len > 65535u32 {
ata_scsi_set_invalid_field(dev, scmd, 10u16, 0u8);
return 1;
}
unsafe {
(*tf).protocol = 2;
}
unsafe {
(*tf).__slate_anon_14.command = 18;
}
unsafe {
(*tf).hob_feature = *cdb.wrapping_offset(14 as isize);
}
unsafe {
(*tf).hob_nsect = (len >> 8i32 & 255) as u8;
}
unsafe {
(*tf).nsect = (len & 255) as u8;
}
starting_element = get_unaligned_be32(cdb.wrapping_offset(6 as isize) as *const ());
if starting_element != 0 {
unsafe {
(*tf).hob_lbal = (starting_element >> 24i32 & 255) as u8;
}
unsafe {
(*tf).lbah = (starting_element >> 16i32 & 255) as u8;
}
unsafe {
(*tf).lbam = (starting_element >> 8i32 & 255) as u8;
}
unsafe {
(*tf).lbal = (starting_element & 255) as u8;
}
}
unsafe {
(*tf).device = 64;
}
unsafe {
(*tf).flags = (*tf).flags | 7u64;
}
ata_qc_set_pc_nbytes(qc);
unsafe {
(*qc).flags = (*qc).flags | 16;
}
unsafe {
(*qc).complete_fn = Some(ata_scsi_get_phys_element_status_complete);
}
0
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_remove_element_and_truncate_xlat")]
extern "C" fn ata_scsi_remove_element_and_truncate_xlat(qc: *mut ata_queued_cmd) -> u32 {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let cdb: *const u8 = (unsafe { (*scmd).cmnd.as_mut_ptr() as *mut u8 }) as *const u8;
let dev: *mut ata_device = unsafe { (*qc).dev };
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).tf) };
let mut req_capacity: u64 = 0;
let mut id: u32 = 0;
if (unsafe { (*dev).flags }) & 16384 == 0 {
ata_scsi_set_sense(dev, scmd, 5u8, 32u8, 0u8);
return 1;
}
req_capacity = get_unaligned_be64(cdb.wrapping_offset(2 as isize) as *const ());
if req_capacity == 1 {
ata_scsi_set_invalid_field(dev, scmd, 2u16, 0u8);
return 1;
}
id = get_unaligned_be32(cdb.wrapping_offset(10 as isize) as *const ());
unsafe {
(*tf).protocol = 0;
}
unsafe {
(*tf).__slate_anon_14.command = 124;
}
unsafe {
(*tf).hob_feature = (id >> 24i32 & 255) as u8;
}
unsafe {
(*tf).__slate_anon_8.feature = (id >> 16i32 & 255) as u8;
}
unsafe {
(*tf).hob_nsect = (id >> 8i32 & 255) as u8;
}
unsafe {
(*tf).nsect = (id & 255) as u8;
}
unsafe {
(*tf).hob_lbah = (req_capacity >> 40i32 & 255) as u8;
}
unsafe {
(*tf).hob_lbam = (req_capacity >> 32i32 & 255) as u8;
}
unsafe {
(*tf).hob_lbal = (req_capacity >> 24i32 & 255) as u8;
}
unsafe {
(*tf).lbah = (req_capacity >> 16i32 & 255) as u8;
}
unsafe {
(*tf).lbam = (req_capacity >> 8i32 & 255) as u8;
}
unsafe {
(*tf).lbal = (req_capacity & 255) as u8;
}
unsafe {
(*tf).device = 64;
}
unsafe {
(*tf).flags = (*tf).flags | 7u64;
}
unsafe {
(*qc).flags = (*qc).flags | 16;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_remove_element_and_modify_zones_xlat")]
extern "C" fn ata_scsi_remove_element_and_modify_zones_xlat(qc: *mut ata_queued_cmd) -> u32 {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let cdb: *const u8 = (unsafe { (*scmd).cmnd.as_mut_ptr() as *mut u8 }) as *const u8;
let dev: *mut ata_device = unsafe { (*qc).dev };
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).tf) };
let mut id: u32 = 0;
if (unsafe { (*dev).flags }) & 65536 == 0 {
ata_scsi_set_sense(dev, scmd, 5u8, 32u8, 0u8);
return 1;
}
id = get_unaligned_be32(cdb.wrapping_offset(10 as isize) as *const ());
unsafe {
(*tf).protocol = 0;
}
unsafe {
(*tf).__slate_anon_14.command = 126;
}
unsafe {
(*tf).hob_feature = (id >> 24i32 & 255) as u8;
}
unsafe {
(*tf).__slate_anon_8.feature = (id >> 16i32 & 255) as u8;
}
unsafe {
(*tf).hob_nsect = (id >> 8i32 & 255) as u8;
}
unsafe {
(*tf).nsect = (id & 255) as u8;
}
unsafe {
(*tf).device = 64;
}
unsafe {
(*tf).flags = (*tf).flags | 7u64;
}
unsafe {
(*qc).flags = (*qc).flags | 16;
}
0
}

#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_restore_elements_and_rebuild_xlat")]
extern "C" fn ata_scsi_restore_elements_and_rebuild_xlat(qc: *mut ata_queued_cmd) -> u32 {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let dev: *mut ata_device = unsafe { (*qc).dev };
let tf: *mut ata_taskfile = unsafe { core::ptr::addr_of_mut!((*qc).tf) };
if (unsafe { (*dev).flags }) & 32768 == 0 {
ata_scsi_set_sense(dev, scmd, 5u8, 32u8, 0u8);
return 1;
}
unsafe {
(*tf).protocol = 0;
}
unsafe {
(*tf).__slate_anon_14.command = 125;
}
unsafe {
(*tf).device = 64;
}
unsafe {
(*tf).flags = (*tf).flags | 7u64;
}
unsafe {
(*qc).flags = (*qc).flags | 16;
}
0
}

/// ata_scsi_var_len_cdb_xlat - SATL variable length CDB to Handler
/// @qc: Command to be translated
///
/// Translate a SCSI variable length CDB to specified commands.
/// It checks a service action value in CDB to call corresponding handler.
///
/// RETURNS:
/// Zero on success, non-zero on failure
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_var_len_cdb_xlat")]
extern "C" fn ata_scsi_var_len_cdb_xlat(qc: *mut ata_queued_cmd) -> u32 {
let scmd: *mut scsi_cmnd = unsafe { (*qc).scsicmd };
let cdb: *const u8 = (unsafe { (*scmd).cmnd.as_mut_ptr() as *mut u8 }) as *const u8;
let sa: u16 = get_unaligned_be16(cdb.wrapping_offset(8 as isize) as *const ());
// if service action represents a ata pass-thru(32) command,
// then pass it to ata_scsi_pass_thru handler.
if (sa as i32) == 8176 {
return ata_scsi_pass_thru(qc);
}
// unsupported service action
1
}

// ata_get_xlat_func - check if SCSI to ATA translation is possible
// @dev: ATA device
// @cdb: CDB of the SCSI command to consider
//
// Look up the SCSI command given, and determine whether the
// SCSI command is to be translated or simulated.
//
// RETURNS:
// Pointer to translation function if possible, %NULL if not.
fn ata_get_xlat_func(dev: *mut ata_device, cdb: *mut u8) -> Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32> {
let mut sa: u8 = 0;
match (unsafe { *cdb.wrapping_offset(0 as isize) }) as i32 {
8 | 10 | 40 | 42 | 136 | 138 => {
return Some(ata_scsi_rw_xlat);
}
147 => {
return Some(ata_scsi_write_same_xlat);
}
53 | 145 => {
if ata_try_flush_cache(dev as *const ata_device) != 0 {
return Some(ata_scsi_flush_xlat);
}
}
47 | 143 => {
return Some(ata_scsi_verify_xlat);
}
133 | 161 => {
return Some(ata_scsi_pass_thru);
}
127 => {
return Some(ata_scsi_var_len_cdb_xlat);
}
21 | 85 => {
return Some(ata_scsi_mode_select_xlat);
}
158 => {
sa = (((unsafe { *cdb.wrapping_offset(1 as isize) }) as i32) & 31) as u8;
if (sa as i32) == 23 {
return Some(ata_scsi_get_phys_element_status_xlat);
}
if (sa as i32) == 24 {
return Some(ata_scsi_remove_element_and_truncate_xlat);
}
if (sa as i32) == 26 {
return Some(ata_scsi_remove_element_and_modify_zones_xlat);
}
if (sa as i32) == 25 {
return Some(ata_scsi_restore_elements_and_rebuild_xlat);
}
}
149 => {
return Some(ata_scsi_zbc_in_xlat);
}
148 => {
return Some(ata_scsi_zbc_out_xlat);
}
162 | 181 => {
if (unsafe { (*dev).flags }) & 256 != 0 {
return Some(ata_scsi_security_inout_xlat);
}
}
27 => {
return Some(ata_scsi_start_stop_xlat);
}
_ => {
}
}
None
}

/// ata_scsi_simulate - simulate SCSI command on ATA device
/// @dev: the target device
/// @cmd: SCSI command being sent to device.
///
/// Interprets and directly executes a select list of SCSI commands
/// that can be handled internally.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
fn ata_scsi_simulate(dev: *mut ata_device, cmd: *mut scsi_cmnd) {
let scsicmd: *const u8 = (unsafe { (*cmd).cmnd.as_mut_ptr() as *mut u8 }) as *const u8;
let mut tmp8: u8 = 0;
'__slate_break_0: {
match (unsafe { *scsicmd.wrapping_offset(0 as isize) }) as i32 {
18 => {
ata_scsi_rbuf_fill(dev, cmd, Some(ata_scsiop_inquiry));
}
26 | 90 => {
ata_scsi_rbuf_fill(dev, cmd, Some(ata_scsiop_mode_sense));
}
37 | 158 => {
ata_scsi_rbuf_fill(dev, cmd, Some(ata_scsiop_read_cap));
}
160 => {
ata_scsi_rbuf_fill(dev, cmd, Some(ata_scsiop_report_luns));
}
3 => {
ata_scsi_set_sense(dev, cmd, 0u8, 0u8, 0u8);
break '__slate_break_0;
// if we reach this, then writeback caching is disabled,
// turning this into a no-op.
}
0 | 1 | 11 | 43 | 53 | 145 => {
// no-op's, complete with success
}
29 => {
tmp8 = (((unsafe { *scsicmd.wrapping_offset(1 as isize) }) as i32) & -9i32) as u8;
if (tmp8 as i32) != 4 || (unsafe { *scsicmd.wrapping_offset(3 as isize) }) != 0 || (unsafe { *scsicmd.wrapping_offset(4 as isize) }) != 0 {
ata_scsi_set_invalid_field(dev, cmd, 1u16, 255u8);
}
}
163 => {
ata_scsi_rbuf_fill(dev, cmd, Some(ata_scsiop_maint_in));
break '__slate_break_0;
// all other commands
}
_ => {
ata_scsi_set_sense(dev, cmd, 5u8, 32u8, 0u8);
// "Invalid command operation code"
}
}
}
unsafe { scsi_done(cmd) };
}

#[unsafe(no_mangle)]
extern "C" fn __ata_scsi_queuecmd(scmd: *mut scsi_cmnd, dev: *mut ata_device, ap: *mut ata_port) -> u32 {
let mut xlat_func: core::mem::MaybeUninit<Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32> = core::ptr::addr_of_mut!(xlat_func) as *mut Option<unsafe extern "C" fn(*mut ata_queued_cmd) -> u32>;
let mut scsi_op: u8 = unsafe { core::mem::zeroed() };
let mut cdb: *mut u8 = unsafe { core::mem::zeroed() };
unsafe {
cdb = (*scmd).cmnd.as_mut_ptr() as *mut u8;
scsi_op = *cdb.wrapping_offset(0 as isize);
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
// scsi_queue_rq() will defer commands if scsi_host_in_recovery().
// However, this check is done without holding the ap->lock (a libata
// specific lock), so we can have received an error irq since then,
// therefore we must check if EH is pending or running, while holding
// ap->lock.
if ata_port_eh_scheduled(ap) {
return 4182;
} else {
'__join_0: {
if unlikely!((*scmd).cmd_len == 0) == 0 {
if (*dev).class == 1 || (*dev).class == 9 {
if unlikely!((((*scmd).cmd_len as i32) as u32) > (*dev).cdb_len) != 0 {
break '__join_0;
} else {
*__slate_slot_1 = ata_get_xlat_func(dev, cdb);
}
} else {
if likely!((scsi_op as i32) != 133 || atapi_passthru16 == 0) != 0 {
// relay SCSI command to ATAPI device
let len: i32 = COMMAND_SIZE!(scsi_op) as i32;
if unlikely!(len > ((*scmd).cmd_len as i32) || (len as u32) > (*dev).cdb_len || ((*scmd).cmd_len as i32) > 16) != 0 {
break '__join_0;
} else {
*__slate_slot_1 = Some(atapi_xlat);
}
} else {
// ATA_16 passthru, treat as an ATA command
if unlikely!(((*scmd).cmd_len as i32) > 16) != 0 {
break '__join_0;
} else {
*__slate_slot_1 = ata_get_xlat_func(dev, cdb);
}
}
}
if *__slate_slot_1 != None {
return ata_scsi_translate(dev, scmd, *__slate_slot_1, ap) as u32;
} else {
ata_scsi_simulate(dev, scmd);
return 0;
}
}
}
(*scmd).result = 458752i32;
scsi_done(scmd);
return 0;
}
}
}

/// ata_scsi_queuecmd - Issue SCSI cdb to libata-managed device
/// @shost: SCSI host of command to be sent
/// @cmd: SCSI command to be sent
///
/// In some cases, this function translates SCSI commands into
/// ATA taskfiles, and queues the taskfiles to be sent to
/// hardware.  In other cases, this function simulates a
/// SCSI device by evaluating and responding to certain
/// SCSI commands.  This creates the overall effect of
/// ATA and ATAPI devices appearing as SCSI devices.
///
/// LOCKING:
/// ATA host lock
///
/// RETURNS:
/// Return value from __ata_scsi_queuecmd() if @cmd can be queued,
/// 0 otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_queuecmd")]
extern "C" fn ata_scsi_queuecmd(shost: *mut Scsi_Host, cmd: *mut scsi_cmnd) -> u32 {
let scsidev: *mut scsi_device = unsafe { (*cmd).device };
let mut rc: u32 = 0;
let mut irq_flags: u64 = 0;
let ap: *mut ata_port = ata_shost_to_port(shost);
spin_lock_irqsave!(unsafe { (*ap).lock }, irq_flags);
let dev: *mut ata_device = ata_scsi_find_dev(ap, scsidev as *const scsi_device);
if likely__5!(dev) != 0 {
rc = __ata_scsi_queuecmd(cmd, dev, ap);
} else {
unsafe {
(*cmd).result = 262144i32;
}
unsafe { scsi_done(cmd) };
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, irq_flags);
rc
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ata_scsi_queuecmd_731: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut Scsi_Host, *mut scsi_cmnd) -> u32>, *mut ()>(Some(ata_scsi_queuecmd)) };

#[unsafe(no_mangle)]
extern "C" fn ata_scsi_add_hosts(host: *mut ata_host, sht: *const scsi_host_template) -> i32 {
let mut shost_1: core::mem::MaybeUninit<*mut Scsi_Host> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut Scsi_Host = core::ptr::addr_of_mut!(shost_1) as *mut *mut Scsi_Host;
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
let mut rc: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(rc) as *mut i32;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
*__slate_slot_8 = 0;
'__join_1: {
while (*__slate_slot_8 as u32) < (*host).n_ports {
ap = *(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(*__slate_slot_8 as isize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_7 = 12i32.wrapping_neg();
*__slate_slot_5 = scsi_host_alloc(sht, core::mem::size_of::<*mut ata_port>() as i32);
if (*__slate_slot_5).is_null() {
break '__join_1;
} else {
(*(*__slate_slot_5)).eh_noresume = true;
*((core::ptr::addr_of_mut!((*(*__slate_slot_5)).hostdata) as *mut u64).wrapping_offset(0 as isize) as *mut *mut ata_port) = ap;
(*ap).scsi_host = *__slate_slot_5;
(*(*__slate_slot_5)).transportt = core::ptr::addr_of_mut!(ata_scsi_transportt);
(*(*__slate_slot_5)).unique_id = (*ap).print_id;
(*(*__slate_slot_5)).max_id = 16;
(*(*__slate_slot_5)).max_lun = 1;
(*(*__slate_slot_5)).max_channel = 1;
(*(*__slate_slot_5)).max_cmd_len = 32;
// Schedule policy is determined by ->qc_defer()
// callback and it needs to see every deferred qc.
// Set host_blocked to 1 to prevent SCSI midlayer from
// automatically deferring requests.
(*(*__slate_slot_5)).max_host_blocked = 1;
*__slate_slot_7 = scsi_add_host_with_dma(*__slate_slot_5, core::ptr::addr_of_mut!((*ap).tdev), (*(*ap).host).dev);
if *__slate_slot_7 != 0 {
break '__join_1;
} else {
let __v4: i32 = *__slate_slot_8;
*__slate_slot_8 = __v4.wrapping_add(1);
}
}
}
return 0;
}
loop {
let __v1: i32 = *__slate_slot_8;
let __v0: i32 = __v1.wrapping_sub(1);
*__slate_slot_8 = __v0;
if __v0 >= 0 {
let shost: *mut Scsi_Host = (*(*(core::ptr::addr_of_mut!((*host).ports) as *mut *mut ata_port).wrapping_offset(*__slate_slot_8 as isize))).scsi_host;
// scsi_host_put() is in ata_devres_release()
scsi_remove_host(shost);
} else {
break;
}
}
return *__slate_slot_7;
}
}

fn ata_scsi_assign_ofnode(dev: *mut ata_device, ap: *mut ata_port) {
}

#[unsafe(no_mangle)]
extern "C" fn ata_scsi_scan_host(ap: *mut ata_port, sync: i32) {
let mut id: i32 = unsafe { core::mem::zeroed() };
let mut channel: i32 = unsafe { core::mem::zeroed() };
let mut sdev: core::mem::MaybeUninit<*mut scsi_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut *mut scsi_device = core::ptr::addr_of_mut!(sdev) as *mut *mut scsi_device;
let mut dev: core::mem::MaybeUninit<*mut ata_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut *mut ata_device = core::ptr::addr_of_mut!(dev) as *mut *mut ata_device;
let mut link: core::mem::MaybeUninit<*mut ata_link> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ata_link = core::ptr::addr_of_mut!(link) as *mut *mut ata_link;
let mut last_failed_dev: *mut ata_device = unsafe { core::mem::zeroed() };
let mut tries: i32 = unsafe { core::mem::zeroed() };
unsafe {
tries = 5;
last_failed_dev = core::ptr::null_mut::<ata_device>();
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
'__join_0: {
loop {
*__slate_slot_7 = ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32);
while !(*__slate_slot_7).is_null() {
*__slate_slot_6 = ata_dev_next(core::ptr::null_mut::<ata_device>(), *__slate_slot_7, 0u32);
while !(*__slate_slot_6).is_null() {
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
channel = 0;
id = 0;
if unsafe { (*(*__slate_slot_6)).sdev }.is_null() {
if ata_is_host_link(*__slate_slot_7 as *const ata_link) {
id = (*(*__slate_slot_6)).devno as i32;
} else {
channel = (*(*__slate_slot_7)).pmp;
}
*__slate_slot_5 = __scsi_add_device((*ap).scsi_host, channel as u32, id as u32, 0u64, core::ptr::null_mut::<()>());
if !IS_ERR(*__slate_slot_5 as *const ()) {
(*(*__slate_slot_6)).sdev = *__slate_slot_5;
ata_scsi_assign_ofnode(*__slate_slot_6, ap);
scsi_device_put(*__slate_slot_5);
} else {
(*(*__slate_slot_6)).sdev = core::ptr::null_mut::<scsi_device>();
}
}
*__slate_slot_6 = ata_dev_next(*__slate_slot_6, *__slate_slot_7, 0u32);
}
*__slate_slot_7 = ata_link_next(*__slate_slot_7, ap, 0u32);
}
// If we scanned while EH was in progress or allocation
// failure occurred, scan would have failed silently.  Check
// whether all devices are attached.
*__slate_slot_7 = ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32);
'__loop_10: while !(*__slate_slot_7).is_null() {
*__slate_slot_6 = ata_dev_next(core::ptr::null_mut::<ata_device>(), *__slate_slot_7, 0u32);
while !(*__slate_slot_6).is_null() {
if unsafe { (*(*__slate_slot_6)).sdev }.is_null() {
break '__loop_10;
} else {
*__slate_slot_6 = ata_dev_next(*__slate_slot_6, *__slate_slot_7, 0u32);
}
}
*__slate_slot_7 = ata_link_next(*__slate_slot_7, ap, 0u32);
}
if (*__slate_slot_7).is_null() {
return;
} else {
// we're missing some SCSI devices
if sync != 0 {
// If caller requested synchrnous scan && we've made
// any progress, sleep briefly and repeat.
if *__slate_slot_6 != last_failed_dev {
msleep(100u32);
last_failed_dev = *__slate_slot_6;
} else {
// We might be failing to detect boot device, give it
// a few more chances.
let __v1: i32 = tries.wrapping_sub(1);
tries = __v1;
if __v1 != 0 {
msleep(100u32);
} else {
break;
}
}
} else {
break '__join_0;
}
}
}
while false {
}
_printk(c"\x013ata%u: WARNING: synchronous SCSI scan failed without making any progress, switching to async\n".as_ptr() as *const u8, (*ap).print_id);
}
queue_delayed_work(system_dfl_long_wq, core::ptr::addr_of_mut!((*ap).hotplug_task), round_jiffies_relative(1000u64));
}
}

/// ata_scsi_offline_dev - offline attached SCSI device
/// @dev: ATA device to offline attached SCSI device for
///
/// This function is called from ata_eh_detach_dev() and is responsible for
/// taking the SCSI device attached to @dev offline.  This function is
/// called with host lock which protects dev->sdev against clearing.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
///
/// RETURNS:
/// true if attached SCSI device exists, false otherwise.
#[unsafe(no_mangle)]
extern "C" fn ata_scsi_offline_dev(dev: *mut ata_device) -> bool {
if !unsafe { (*dev).sdev }.is_null() {
unsafe { scsi_device_set_state((*dev).sdev, 6u32) };
return true;
}
false
}

/// ata_scsi_remove_dev - remove attached SCSI device
/// @dev: ATA device to remove attached SCSI device for
///
/// This function is called from ata_eh_scsi_hotplug() and
/// responsible for removing the SCSI device attached to @dev.
///
/// LOCKING:
/// Kernel thread context (may sleep).
fn ata_scsi_remove_dev(dev: *mut ata_device) {
let ap: *mut ata_port = unsafe { (*(*dev).link).ap };
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
// Alas, we need to grab scan_mutex to ensure SCSI device
// state doesn't change underneath us and thus
// scsi_device_get() always succeeds.  The mutex locking can
// be removed if there is __scsi_device_get() interface which
// increments reference counts regardless of device state.
unsafe { mutex_lock(core::ptr::addr_of_mut!((*(*ap).scsi_host).scan_mutex)) };
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
// clearing dev->sdev is protected by host lock
sdev = unsafe { (*dev).sdev };
unsafe {
(*dev).sdev = core::ptr::null_mut::<scsi_device>();
}
if !sdev.is_null() {
// If user initiated unplug races with us, sdev can go
// away underneath us after the host lock and
// scan_mutex are released.  Hold onto it.
if (unsafe { scsi_device_get(sdev) }) == 0 {
// The following ensures the attached sdev is
// offline on return from ata_scsi_offline_dev()
// regardless it wins or loses the race
// against this function.
unsafe { scsi_device_set_state(sdev, 6u32) };
} else {
let mut __v3: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("732: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 732b - ., 3; .popsection\n# {0}", const 732 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ata_scsi_remove_dev_4, sym __slate_asm_string_ata_scsi_remove_dev_5, const 5428 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("733: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 733b - ., 4; .popsection\n# {0}", const 733 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
sdev = core::ptr::null_mut::<scsi_device>();
}
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*(*ap).scsi_host).scan_mutex)) };
if !sdev.is_null() {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x016ata%u.%02u: detaching (SCSI %s)\n".as_ptr() as *const u8, (*(*(*dev).link).ap).print_id, ((*(*dev).link).pmp as u32).wrapping_add((*dev).devno), dev_name(core::ptr::addr_of_mut!((*sdev).sdev_gendev) as *const device)) };
unsafe { scsi_remove_device(sdev) };
unsafe { scsi_device_put(sdev) };
}
}

fn ata_scsi_handle_link_detach(link: *mut ata_link) {
let ap: *mut ata_port = unsafe { (*link).ap };
let mut dev: *mut ata_device = unsafe { core::mem::zeroed() };
dev = unsafe { ata_dev_next(core::ptr::null_mut::<ata_device>(), link, 2u32) };
while !dev.is_null() {
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
if (unsafe { (*dev).flags }) & 134217728 == 0 {
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
} else {
let __v6: u64 = (unsafe { (*dev).flags }) & 4160749567u64;
unsafe {
(*dev).flags = __v6;
}
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
ata_scsi_remove_dev(dev);
}
dev = unsafe { ata_dev_next(dev, link, 2u32) };
}
}

/// ata_scsi_media_change_notify - send media change event
/// @dev: Pointer to the disk device with media change event
///
/// Tell the block layer to send a media change notification
/// event.
///
/// LOCKING:
/// spin_lock_irqsave(host lock)
#[unsafe(no_mangle)]
extern "C" fn ata_scsi_media_change_notify(dev: *mut ata_device) {
if !unsafe { (*dev).sdev }.is_null() {
unsafe { sdev_evt_send_simple((*dev).sdev, 1u32, (BIT!(5i32) as u32) | (BIT!(11i32) as u32)) };
}
}

/// ata_scsi_hotplug - SCSI part of hotplug
/// @work: Pointer to ATA port to perform SCSI hotplug on
///
/// Perform SCSI part of hotplug.  It's executed from a separate
/// workqueue after EH completes.  This is necessary because SCSI
/// hot plugging requires working EH and hot unplugging is
/// synchronized with hot plugging with a mutex.
///
/// LOCKING:
/// Kernel thread context (may sleep).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_hotplug")]
extern "C" fn ata_scsi_hotplug(work: *mut work_struct) {
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
let mut __v0: *mut ata_port;
__v0 = ((unsafe { (work as *const u8).wrapping_offset(-(offsetof__150!(ata_port) as isize)) }) as *mut ()) as *mut ata_port;
ap = __v0;
if (unsafe { (*ap).pflags }) & 512 != 0 {
return;
}
unsafe { mutex_lock(core::ptr::addr_of_mut!((*ap).scsi_scan_mutex)) };
// Unplug detached devices.  We cannot use link iterator here
// because PMP links have to be scanned even if PMP is
// currently not attached.  Iterate manually.
ata_scsi_handle_link_detach(unsafe { core::ptr::addr_of_mut!((*ap).link) });
if !unsafe { (*ap).pmp_link }.is_null() {
for i in 0i32..15i32 {
ata_scsi_handle_link_detach(unsafe { (*ap).pmp_link }.wrapping_offset(i as isize));
}
}
// scan for new ones
ata_scsi_scan_host(ap, 0i32);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*ap).scsi_scan_mutex)) };
}

/// ata_scsi_user_scan - indication for user-initiated bus scan
/// @shost: SCSI host to scan
/// @channel: Channel to scan
/// @id: ID to scan
/// @lun: LUN to scan
///
/// This function is called when user explicitly requests bus
/// scan.  Set probe pending flag and invoke EH.
///
/// LOCKING:
/// SCSI layer (we don't care)
///
/// RETURNS:
/// Zero.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_user_scan")]
extern "C" fn ata_scsi_user_scan(shost: *mut Scsi_Host, channel: u32, id: u32, lun: u64) -> i32 {
let ap: *mut ata_port = ata_shost_to_port(shost);
let mut flags: u64 = 0;
let mut devno: i32 = 0;
let mut rc: i32 = 0;
if lun != 18446744073709551615u64 && lun != 0 {
return 22i32.wrapping_neg();
}
if !sata_pmp_attached(ap) {
if channel != 4294967295u32 && channel != 0 {
return 22i32.wrapping_neg();
}
devno = id as i32;
} else {
if id != 4294967295u32 && id != 0 {
return 22i32.wrapping_neg();
}
devno = channel as i32;
}
spin_lock_irqsave!(unsafe { (*ap).lock }, flags);
if devno == !0 {
let mut link: *mut ata_link = unsafe { core::mem::zeroed() };
link = unsafe { ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32) };
while !link.is_null() {
let ehi: *mut ata_eh_info = unsafe { core::ptr::addr_of_mut!((*link).eh_info) };
unsafe {
(*ehi).probe_mask = (*ehi).probe_mask | 3;
}
unsafe {
(*ehi).action = (*ehi).action | 6;
}
link = unsafe { ata_link_next(link, ap, 0u32) };
}
} else {
let dev: *mut ata_device = ata_find_dev(ap, devno as u32);
if !dev.is_null() {
let ehi: *mut ata_eh_info = unsafe { core::ptr::addr_of_mut!((*(*dev).link).eh_info) };
unsafe {
(*ehi).probe_mask = (*ehi).probe_mask | ((1i32 << (*dev).devno) as u32);
}
unsafe {
(*ehi).action = (*ehi).action | 6;
}
} else {
rc = 22i32.wrapping_neg();
}
}
if rc == 0 {
unsafe { ata_port_schedule_eh(ap) };
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
unsafe { ata_port_wait_eh(ap) };
} else {
spin_unlock_irqrestore(unsafe { (*ap).lock }, flags);
}
rc
}

/// ata_scsi_dev_rescan - initiate scsi_rescan_device()
/// @work: Pointer to ATA port to perform scsi_rescan_device()
///
/// After ATA pass thru (SAT) commands are executed successfully,
/// libata need to propagate the changes to SCSI layer.
///
/// LOCKING:
/// Kernel thread context (may sleep).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.libata_scsi.ata_scsi_dev_rescan")]
extern "C" fn ata_scsi_dev_rescan(work: *mut work_struct) {
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut ret: i32 = unsafe { core::mem::zeroed() };
let mut do_resume: core::mem::MaybeUninit<bool> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut bool = core::ptr::addr_of_mut!(do_resume) as *mut bool;
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut dev: core::mem::MaybeUninit<*mut ata_device> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut *mut ata_device = core::ptr::addr_of_mut!(dev) as *mut *mut ata_device;
let mut link: core::mem::MaybeUninit<*mut ata_link> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut *mut ata_link = core::ptr::addr_of_mut!(link) as *mut *mut ata_link;
let mut ap: *mut ata_port = unsafe { core::mem::zeroed() };
unsafe {
ap = ((work as *const u8).wrapping_offset(-(offsetof__120!(ata_port) as isize)) as *mut ()) as *mut ata_port;
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
ret = 0;
mutex_lock(core::ptr::addr_of_mut!((*ap).scsi_scan_mutex));
spin_lock_irqsave!(unsafe { (*ap).lock }, *__slate_slot_6);
*__slate_slot_8 = ata_link_next(core::ptr::null_mut::<ata_link>(), ap, 0u32);
'__join_2: {
'__join_3: {
'__loop_4: loop {
if !(*__slate_slot_8).is_null() {
*__slate_slot_7 = ata_dev_next(core::ptr::null_mut::<ata_device>(), *__slate_slot_8, 0u32);
while !(*__slate_slot_7).is_null() {
sdev = (*(*__slate_slot_7)).sdev;
// If the port was suspended before this was scheduled,
// bail out.
if (*ap).pflags & 131072 != 0 {
break '__join_3;
} else {
if !sdev.is_null() {
if scsi_device_get(sdev) == 0 {
*__slate_slot_5 = (*(*__slate_slot_7)).flags & 33554432 != 0;
spin_unlock_irqrestore((*ap).lock, *__slate_slot_6);
if *__slate_slot_5 {
ret = scsi_resume_device(sdev);
if ret == 11i32.wrapping_neg() {
break '__loop_4;
} else {
let __v2: *mut ata_device = *__slate_slot_7;
let __v0: u64 = (*__v2).flags & 4261412863u64;
(*__v2).flags = __v0;
}
}
ret = scsi_rescan_device(sdev);
scsi_device_put(sdev);
spin_lock_irqsave!(unsafe { (*ap).lock }, *__slate_slot_6);
if ret != 0 {
break '__join_3;
}
}
}
*__slate_slot_7 = ata_dev_next(*__slate_slot_7, *__slate_slot_8, 0u32);
}
}
*__slate_slot_8 = ata_link_next(*__slate_slot_8, ap, 0u32);
} else {
break '__join_3;
}
}
scsi_device_put(sdev);
break '__join_2;
}
spin_unlock_irqrestore((*ap).lock, *__slate_slot_6);
}
mutex_unlock(core::ptr::addr_of_mut!((*ap).scsi_scan_mutex));
// Reschedule with a delay if scsi_rescan_device() returned an error
if ret != 0 {
schedule_delayed_work(core::ptr::addr_of_mut!((*ap).scsi_rescan_task), msecs_to_jiffies(5u32));
}
}
}

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_ata_scsi_deferred_qc_work_6: [u8; 1] = *b"\0";

static __slate_asm_string_ata_scsi_deferred_qc_work_7: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-scsi.c\0";

static __slate_asm_string_ata_scsi_qc_issue_5: [u8; 1] = *b"\0";

static __slate_asm_string_ata_scsi_qc_issue_6: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-scsi.c\0";

static __slate_asm_string_ata_scsi_qc_new_10: [u8; 1] = *b"\0";

static __slate_asm_string_ata_scsi_qc_new_11: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-scsi.c\0";

static __slate_asm_string_ata_scsi_rbuf_fill_4: [u8; 1] = *b"\0";

static __slate_asm_string_ata_scsi_rbuf_fill_5: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-scsi.c\0";

static __slate_asm_string_ata_scsi_remove_dev_4: [u8; 1] = *b"\0";

static __slate_asm_string_ata_scsi_remove_dev_5: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-scsi.c\0";

static __slate_asm_string_ata_scsi_report_zones_complete_5: [u8; 1] = *b"\0";

static __slate_asm_string_ata_scsi_report_zones_complete_6: [u8; 55] = *b"/home/takashi/c-corpus/linux/drivers/ata/libata-scsi.c\0";

static __slate_asm_string_check_copy_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_check_copy_size_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/include/linux/ucopysize.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

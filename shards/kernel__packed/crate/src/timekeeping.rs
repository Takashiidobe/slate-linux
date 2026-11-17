
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
//! SPDX-License-Identifier: GPL-2.0
//!
//! Kernel timekeeping code and accessor functions. Based on code from
//! timer.c, moved in commit 8524070b7982.
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
//! New user namespace
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
//! LLVM integrated assembler cam merge adjacent string literals (like
//! C and GNU-as) passed to '.ascii', but not to '.asciz' and chokes on:
//!
//!   .asciz "MODULE_" "kvm" ;
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
//! warning conditions
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
//! Helper macro for scoped_guard().
//!
//! Note that the "!__is_cond_ptr(_name)" part of the condition ensures that
//! compiler would be sure that for the unconditional locks the body of the
//! loop (caller-provided code glued to the else clause) could not be skipped.
//! It is needed because the other part - "__guard_ptr(_name)(&scope)" - is too
//! hard to deduce (even if could be proven true for unconditional locks).
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
//! Physical address zone modifiers (see linux/mmzone.h - low four bits)
//!
//! Do not put any conditional on these. If necessary modify the definitions
//! without the underscores and use them consistently. The definitions here may
//! be used in bit comparisons.
//! container_of - cast a member of a structure out to the containing structure
//! @ptr:	the pointer to the member.
//! @type:	the type of the container struct this is embedded in.
//! @member:	the name of the member within the struct.
//!
//! WARNING: any const qualifier of @ptr is lost.
//! Do not use container_of() in new code.
//! This looks more complex than it should be. But we need to
//! get the type for the ~ right in round_down (it needs to be
//! as wide as the result!), and we want to evaluate the macro
//! arguments just once each.
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
//! Parameters used to convert the timespec values:
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
//! switch between adjtime/adjtimex modes
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
//! GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
//! __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
//! entry starting with bit 0. Bit is set if the combination is not
//! allowed.
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
//! kmem_cache_alloc and friends return pointers aligned to ARCH_SLAB_MINALIGN.
//! kmalloc and friends return pointers aligned to both ARCH_KMALLOC_MINALIGN
//! and ARCH_SLAB_MINALIGN, but here we only assume the former alignment.
//! Bits set in the VMA until the stack is in its final location
//! return the first ptr or next element in an RCU protected list
//! Keep unconverted code working
//! used to be O_SYNC, see below
//! Valid hint values for F_{GET,SET}_RW_HINT. 0 is "not set", or can be
//! used to clear any hints previously set.
//! Gives us 8 prio classes with 13-bits of data for each class
//! per-IO O_SYNC
//! File supports atomic writes
//! number of env pointers
//! modes for vm.memfd_noexec sysctl
//! These values are stored in task->ptrace_message
//! by ptrace_stop to describe the current syscall-stop.
//! SEIZE used, enable new behavior
//! Rule structure sizes -- if these change, different AUDIT_ADD and
//! AUDIT_LIST commands must be implemented.
//! See the comment in asm-generic/io.h about ioremap_np().
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
//! 1.0 as fixed-point
//! Generate SD flag indexes
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
// Note the missing underscores.
//
//   gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-noinline-function-attribute
// clang: mentioned
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
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html#index-weak-function-attribute
// gcc: https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html#index-weak-variable-attribute
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
// SCHED_ISO: reserved but not implemented yet
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
// Atomic operations are already serializing on x86
// The following macros are non-atomic versions of their non-underscored
// counterparts.
// Check that the bitops prototypes are sane
// assign_bit - Assign value to a bit in memory
// @nr: the bit to set
// @addr: the address to start counting from
// @value: the value to assign
// No such file or directory
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// No such device
// Invalid argument
// Text file busy
// No space left on device
// Invalid system call number
// Value too large for defined data type
// Operation not supported on transport endpoint
// Memory page has hardware error
// Operation is not supported
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
// round_up - round up to next specified power of 2
// @x: the value to round
// @y: multiple to round up to (must be a power of 2)
//
// Rounds @x up to next multiple of @y (which must be a power of 2).
// To perform arbitrary rounding up, use roundup() below.
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
// cpumask_of - the cpumask containing just a given cpu
// @cpu: the cpu (<= nr_cpu_ids)
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
// srcu_read_lock().
// srcu_read_lock_nmisafe().
// 					0x4		// SRCU-lite is no longer with us.
// srcu_read_lock_fast(), also NMI-safe.
// srcu_read_lock_fast_updown().
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// Resource extended types
// No address assigned yet
// SLUB directly allocates requests fitting in to an order-1 page
// (PAGE_SIZE*2).  Larger requests are passed to the page allocator.
// Maximum size for which we actually use a slab cache
// Define gfp bits that should not be set for KMALLOC_NORMAL.
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
// PT_TRACE_* event enable flags
// On most systems we can tell if a syscall is a success based on if the retval
// is an error value.  On some systems like ia64 and powerpc they have different
// indicators of success/failure and must define their own.
// Implementation specific max size
// Generate SD flag bits
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
macro_rules! BITS_TO_BYTES {
($nr:expr) => {
__KERNEL_DIV_ROUND_UP!($nr, (core::mem::size_of::<u8>() as u64).wrapping_mul(8))
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

macro_rules! EAX_EDX_VAL {
($low:expr, $high:expr) => {
($low) | ($high) << 32i32
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
unlikely__2!(((($x) as *mut ()) as u64) >= (4095i32.wrapping_neg() as u64))
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

macro_rules! __KERNEL_DIV_ROUND_UP {
($n:expr, $d:expr) => {
(($n) as u64).wrapping_add($d).wrapping_sub(1) / $d
};
}

macro_rules! __READ_ONCE__2 {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i64) }
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

macro_rules! __WRITE_ONCE__11 {
($x:expr, $val:expr) => {
loop {
unsafe { core::ptr::write_volatile(core::ptr::addr_of_mut!($x) as *mut i64, $val) };
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

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__44 {
() => {
((const { core::mem::size_of::<*mut clocksource_base>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut clocksource_base>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut clocksource_base>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut clocksource_base>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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
macro_rules! __native_word__3 {
() => {
((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__57 {
() => {
((const { core::mem::size_of::<*mut clocksource>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut clocksource>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut clocksource>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut clocksource>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
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
macro_rules! __sign_use__3 {
($ux:expr) => {
if is_signed_type__4!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i64>() as u64) < 4) as i32)) }
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
macro_rules! __sign_use__5 {
($ux:expr) => {
2i32.wrapping_add(__is_nonneg!($ux) as i32)
};
}

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use__5!($ux) & __sign_use__5!($uy)
};
}

macro_rules! __types_ok__7 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__2!($uy)
};
}

macro_rules! __types_ok__2 {
($ux:expr, $uy:expr) => {
__sign_use__3!($ux) & __sign_use__3!($uy)
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
let __vpp_verify: *const () = core::ptr::null_mut::<u64>() as *const ();
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

/// abs - return absolute value of an argument
/// @x: the value.
///
/// If it is unsigned type, @x is converted to signed type first.
/// char is treated as if it was signed (regardless of whether it really is)
/// but the macro's return type is preserved as char.
///
/// NOTE, for signed type if @x is the minimum, the returned result is undefined
/// as there is not enough bits to represent it as a positive number.
///
/// Return: an absolute value of @x.
macro_rules! abs__2 {
($x:expr) => {
{
let mut __v: i32;
let __v2: i32 = ($x) as i32;
__v = if __v2 < 0 { __v2.wrapping_neg() } else { __v2 };
__v
}
};
}

/// abs - return absolute value of an argument
/// @x: the value.
///
/// If it is unsigned type, @x is converted to signed type first.
/// char is treated as if it was signed (regardless of whether it really is)
/// but the macro's return type is preserved as char.
///
/// NOTE, for signed type if @x is the minimum, the returned result is undefined
/// as there is not enough bits to represent it as a positive number.
///
/// Return: an absolute value of @x.
macro_rules! abs {
($x:expr) => {
{
let mut __v: i64;
let __v2: i64 = $x;
__v = if __v2 < 0 { __v2.wrapping_neg() } else { __v2 };
__v
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
let __v2: u32 = $expr;
__kcsan_enable_current();
__v = __v2;
__v
}
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
macro_rules! do_div__2 {
($n:expr, $base:expr) => {
let mut __v: u32;
let __base: u32 = $base;
let __rem: u32 = ((($n) as u64) % (__base as u64)) as u32;
$n = (($n) as u64) / (__base as u64);
__v = __rem;
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
macro_rules! ilog2__7 {
($n:expr) => {
__ilog2_u64($n)
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
macro_rules! ilog2__4 {
($n:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($n) } { if ($n) < 2 { 0i32 } else { 63i32.wrapping_sub($n.leading_zeros() as i32) } } else { __ilog2_u64($n) }
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__4 {
() => {
(1i32.wrapping_neg() as i64) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__2 {
() => {
1i32.wrapping_neg() < 1
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

/// kcsan_check_read_write - check regular read-write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_read_write {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 3i32);
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

/// Add two ktime_t variables. res = lhs + rhs:
macro_rules! ktime_add__2 {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_add(($rhs) as u64)
};
}

/// Add two ktime_t variables. res = lhs + rhs:
macro_rules! ktime_add {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_add($rhs)
};
}

/// Add a ktime_t variable and a scalar nanosecond value.
/// res = kt + nsval:
macro_rules! ktime_add_ns {
($kt:expr, $nsval:expr) => {
(($kt) as u64).wrapping_add($nsval)
};
}

/// Subtract two ktime_t variables. rem = lhs -rhs:
macro_rules! ktime_sub {
($lhs:expr, $rhs:expr) => {
$lhs.wrapping_sub($rhs)
};
}

/// Subtract a scalar nanosecod from a ktime_t variable
/// res = kt - nsval:
macro_rules! ktime_sub_ns {
($kt:expr, $nsval:expr) => {
(($kt) as u64).wrapping_sub($nsval)
};
}

/// Map the ktime_t to timespec conversion to ns_to_timespec function
macro_rules! ktime_to_timespec64 {
($kt:expr) => {
unsafe { ns_to_timespec64($kt) }
};
}

macro_rules! likely {
($x:expr) => {
!(!$x) as i64
};
}

macro_rules! lock_acquire {
() => {
while false {
}
};
}

macro_rules! lock_acquire_exclusive {
() => {
lock_acquire!();
};
}

macro_rules! lock_release {
() => {
while false {
}
};
}

macro_rules! lockdep_assert_held {
($l:expr) => {
__assume_ctx_lock!($l);
};
}

macro_rules! lockdep_assert_preemption_disabled {
() => {
while false {
}
};
}

macro_rules! lockdep_init_map {
($name:expr, $key:expr) => {
loop {
$name;
$key;
if true {
break;
}
}
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

/// read_seqcount_begin() - begin a seqcount_t read critical section
/// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
///
/// Return: count to be passed to read_seqcount_retry()
macro_rules! read_seqcount_begin {
($s:expr) => {
{
let mut __v: u32;
let mut __v2: u32;
let mut __v3: u32 = 0;
loop {
let __v4: u32 = seqprop_sequence!($s);
__v3 = __v4;
if __v4 & 1 == 0 {
break;
}
cpu_relax();
}
kcsan_atomic_next(1000i32);
__v2 = __v3;
__v = __v2;
__v
}
};
}

/// read_seqcount_retry() - end a seqcount_t read critical section
/// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
/// @start: count, from read_seqcount_begin()
///
/// read_seqcount_retry closes the read critical section of given
/// seqcount_t.  If the critical section was invalid, it must be ignored
/// (and typically retried).
///
/// Return: true if a read section retry is required, else false
macro_rules! read_seqcount_retry {
($s:expr, $start:expr) => {
do_read_seqcount_retry(seqprop_const_ptr!($s), $start)
};
}

macro_rules! seqcount_acquire {
() => {
lock_acquire_exclusive!();
};
}

macro_rules! seqcount_init {
($s:expr) => {
__seqcount_init($s, core::ptr::null::<u8>(), core::ptr::null_mut::<lock_class_key>());
};
}

macro_rules! seqcount_raw_spinlock_init {
($s:expr) => {
loop {
let ____s: *mut seqcount_raw_spinlock = $s;
seqcount_init!(unsafe { core::ptr::addr_of_mut!((*____s).seqcount) });
if true {
break;
}
}
};
}

macro_rules! seqcount_release {
() => {
lock_release!();
};
}

macro_rules! seqprop_assert {
($s:expr) => {
__seqprop_raw_spinlock_assert(($s) as *const seqcount_raw_spinlock);
};
}

macro_rules! seqprop_const_ptr {
($s:expr) => {
__seqprop_raw_spinlock_const_ptr(($s) as *const seqcount_raw_spinlock)
};
}

macro_rules! seqprop_preemptible {
($s:expr) => {
__seqprop_raw_spinlock_preemptible(($s) as *const seqcount_raw_spinlock)
};
}

macro_rules! seqprop_ptr {
($s:expr) => {
__seqprop_raw_spinlock_ptr($s)
};
}

macro_rules! seqprop_sequence {
($s:expr) => {
__seqprop_raw_spinlock_sequence(($s) as *const seqcount_raw_spinlock)
};
}

macro_rules! static_branch_disable {
($x:expr) => {
unsafe { static_key_disable(core::ptr::addr_of_mut!((*$x).key)) };
};
}

macro_rules! static_branch_enable {
($x:expr) => {
unsafe { static_key_enable(core::ptr::addr_of_mut!((*$x).key)) };
};
}

macro_rules! static_branch_likely {
($x:expr) => {
{
let mut __v: i64;
let mut __v2: bool = false;
__v2 = !arch_static_branch_jump__const1(unsafe { core::ptr::addr_of_mut!((*$x).key) }, true);
__v = __v2 as i64;
__v
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

macro_rules! unlikely__6 {
($x:expr) => {
(($x) != None) as i64
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
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut kmsg_fops: file_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut sighand_cachep: *mut kmem_cache;
    static mut sys_tz: timezone;
    static mut jiffies_64: u64;
    static mut timekeeping_mg_floor_swaps: u64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn _printk(arg0: *const u8, ...) -> i32;
    fn _printk_deferred(arg0: *const u8, ...) -> i32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn static_key_enable(arg0: *mut static_key);
    fn static_key_disable(arg0: *mut static_key);
    fn ____wrong_branch_error() -> bool;
    fn __cmpxchg_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn add_device_randomness(arg0: *const (), arg1: u64);
    fn set_normalized_timespec64(arg0: *mut timespec64, arg1: i64, arg2: i64);
    fn ns_to_timespec64(arg0: i64) -> timespec64;
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn __compiletime_assert_195();
    fn __compiletime_assert_196();
    fn __compiletime_assert_197();
    fn __compiletime_assert_198();
    fn __compiletime_assert_199();
    fn __compiletime_assert_200();
    fn __compiletime_assert_201();
    fn __compiletime_assert_202();
    fn __compiletime_assert_203();
    fn __compiletime_assert_204();
    fn __compiletime_assert_205();
    fn __compiletime_assert_206();
    fn __compiletime_assert_207();
    fn __compiletime_assert_208();
    fn __compiletime_assert_209();
    fn __compiletime_assert_210();
    fn __compiletime_assert_211();
    fn __compiletime_assert_212();
    fn __compiletime_assert_213();
    fn __compiletime_assert_214();
    fn __compiletime_assert_215();
    fn raw_notifier_chain_register(arg0: *mut raw_notifier_head, arg1: *mut notifier_block) -> i32;
    fn raw_notifier_chain_unregister(arg0: *mut raw_notifier_head, arg1: *mut notifier_block) -> i32;
    fn raw_notifier_call_chain(arg0: *mut raw_notifier_head, arg1: u64, arg2: *mut ()) -> i32;
    fn capable(arg0: i32) -> bool;
    fn timerfd_resume();
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_344();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn __compiletime_assert_419();
    fn __audit_tk_injoffset(arg0: timespec64);
    fn __audit_ntp_log(arg0: *const audit_ntp_data);
    fn clocksource_suspend();
    fn clocksource_resume();
    fn clocksource_default_clock() -> *mut clocksource;
    fn clocksource_start_suspend_timing(arg0: *mut clocksource, arg1: u64);
    fn clocksource_stop_suspend_timing(arg0: *mut clocksource, arg1: u64) -> u64;
    fn clocks_calc_mult_shift(arg0: *mut u32, arg1: *mut u32, arg2: u32, arg3: u32, arg4: u32);
    fn try_module_get(arg0: *mut module) -> bool;
    fn module_put(arg0: *mut module);
    fn local_clock() -> u64;
    fn calc_global_load();
    fn stop_machine(arg0: Option<unsafe extern "C" fn(*mut ()) -> i32>, arg1: *mut (), arg2: *const cpumask) -> i32;
    fn register_syscore(arg0: *mut syscore);
    fn clockevents_suspend();
    fn clockevents_resume();
    fn tick_suspend();
    fn tick_resume();
    fn tick_clock_notify();
    fn clock_was_set(arg0: u32);
    fn clock_was_set_delayed();
    fn tk_debug_account_sleep_time(arg0: *const timespec64);
    fn update_vsyscall(arg0: *mut timekeeper);
    fn vdso_time_update_aux(arg0: *mut timekeeper);
    fn ntp_init();
    fn ntp_clear(arg0: u32, arg1: i64);
    fn ntp_tick_length(arg0: u32) -> u64;
    fn ntp_get_skew_delta(arg0: u32) -> i64;
    fn ntp_drain_skew(arg0: u32, arg1: i64, arg2: u32) -> i64;
    fn ntp_get_next_leap(arg0: u32) -> i64;
    fn second_overflow(arg0: u32, arg1: i64) -> i32;
    fn ntp_adjtimex(arg0: u32, arg1: *mut __kernel_timex, arg2: *const timespec64, arg3: *mut i32, arg4: *mut audit_ntp_data) -> i32;
    fn ntp_notify_cmos_timer(arg0: bool);
    fn __compiletime_assert_514();
    fn __compiletime_assert_515();
    fn __compiletime_assert_516();
    fn __compiletime_assert_517();
    fn __compiletime_assert_530();
    fn __compiletime_assert_534();
    fn __compiletime_assert_537();
    fn __compiletime_assert_553();
    fn __compiletime_assert_562();
    fn __compiletime_assert_563();
    fn __compiletime_assert_565();
    fn __compiletime_assert_566();
    fn __compiletime_assert_572();
    fn __compiletime_assert_573();
    fn __compiletime_assert_578();
    fn __compiletime_assert_596();
    fn __compiletime_assert_599();
    fn __compiletime_assert_605();
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

fn kcsan_nestable_atomic_begin() {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
}


fn kcsan_nestable_atomic_end() {
}

fn kcsan_atomic_next(n: i32) {
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
fn instrument_read_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_read_write!(v, size);
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
#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn __ilog2_u32(n: u32) -> i32 {
fls(n).wrapping_sub(1)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
}


#[inline(always)]
fn __ilog2_u64(n: u64) -> i32 {
fls64(n).wrapping_sub(1)
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
    __slate_bits_0: __slate_bits::__SlateBits113U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
fn arch_static_branch_jump__const1(key: *mut static_key, branch: bool) -> bool {
unsafe {
let mut __slate_asm_target: bool = false;
core::arch::asm!("1:jmp {2}\n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0} + {1} - . \n\t.popsection \n\t", sym clocksource_read_inlined, const 1 as u8, label {
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
struct pollfd {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_false {
    key: static_key,
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


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct orc_entry {
    sp_offset: i16,
    bp_offset: i16,
    __slate_bits_0: __slate_bits::__SlateBits120U0,
}


#[inline(always)]
fn arch_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
__READ_ONCE__2!((*v).counter)
}

#[inline(always)]
fn arch_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, _old: *mut i64, mut __new: i64) -> bool {
let mut __v0: i64;
let mut success: bool = false;
let mut __old: i64 = unsafe { *_old };
match (const { core::mem::size_of::<i64>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u8;
let __t1: u64;
core::arch::asm!("lock cmpxchgb {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t0, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t1, options(nostack, att_syntax));
success = (__t0 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t1 as u64) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t2: u8;
let __t3: u64;
core::arch::asm!("lock cmpxchgw {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t2, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t3, options(nostack, att_syntax));
success = (__t2 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t3 as u64) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t4: u8;
let __t5: u64;
core::arch::asm!("lock cmpxchgl {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t4, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t5, options(nostack, att_syntax));
success = (__t4 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t5 as u64) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t6: u8;
let __t7: u64;
core::arch::asm!("lock cmpxchgq {2:r}, ({1:r})\nsetz {0}", lateout(reg_byte) __t6, in(reg) __ptr, in(reg) __new as u64, inlateout("ax") __old as u64 => __t7, options(nostack, att_syntax));
success = (__t6 as u8) != 0;
__old = unsafe { core::mem::transmute::<u64, i64>(__t7 as u64) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
if unlikely__2!(!success) != 0 {
unsafe {
*_old = __old;
}
}
__v0 = success as i64;
__v0 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn raw_atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
arch_atomic64_read(v)
}

#[inline(always)]
fn raw_atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
arch_atomic64_try_cmpxchg(v, old, new)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn atomic64_read(v: *const __SlateRecord_types_h_197) -> i64 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic64_read(v)
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
fn atomic64_try_cmpxchg(v: *mut __SlateRecord_types_h_197, old: *mut i64, new: i64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
instrument_read_write(old as *const (), core::mem::size_of::<i64>() as u64);
raw_atomic64_try_cmpxchg(v, old, new)
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
struct futex_private_hash {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_nodemask_types_h_17 {
    bits: [u64; 1],
}


fn mul_u64_add_u64_div_u64(mut rax: u64, mut mul: u64, mut add: u64, mut div: u64) -> u64 {
let mut rdx: u64 = 0;
unsafe {
let __t0: u64;
let __t1: u64;
core::arch::asm!("mulq {0:r}", in(reg) mul as u64, inlateout("ax") rax as u64 => __t0, lateout("dx") __t1, options(pure, nomem, nostack, att_syntax));
rax = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
rdx = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
if !statically_true!(add == 0) {
unsafe {
let __t2: u64;
let __t3: u64;
core::arch::asm!("addq {2:r}, {0:r}; adcq $0, {1:r}", inlateout(reg) rax as u64 => __t2, inlateout(reg) rdx as u64 => __t3, in(reg) add as u64, options(pure, nomem, nostack, att_syntax));
rax = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
rdx = unsafe { core::mem::transmute::<u64, u64>(__t3 as u64) };
}
}
unsafe {
let __t4: u64;
let __t5: u64;
core::arch::asm!("divq {0:r}", in(reg) div as u64, inlateout("ax") rax as u64 => __t4, inlateout("dx") rdx as u64 => __t5, options(pure, nomem, nostack, att_syntax));
rax = unsafe { core::mem::transmute::<u64, u64>(__t4 as u64) };
rdx = unsafe { core::mem::transmute::<u64, u64>(__t5 as u64) };
}
rax
}

fn mul_u64_u32_div(a: u64, mul: u32, div: u32) -> u64 {
mul_u64_add_u64_div_u64(a, mul as u64, 0u64, div as u64)
}

#[inline(always)]
fn native_pause() {
unsafe {
core::arch::asm!("pause", options(nostack, att_syntax));
}
}

#[inline(always)]
fn cpu_relax() {
native_pause();
}

#[inline(always)]
fn __iter_div_u64_rem(mut dividend: u64, divisor: u32, remainder: &mut u64) -> u32 {
let mut ret: u32 = 0;
while dividend >= (divisor as u64) {
// The following asm() prevents the compiler from
// optimising this loop into a modulo operation.
unsafe {
let __t0: u64;
core::arch::asm!("\n# {0:r}", inlateout(reg) dividend as u64 => __t0, options(pure, nomem, nostack, att_syntax));
dividend = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
dividend = dividend.wrapping_sub(divisor as u64);
ret = ret.wrapping_add(1);
}
unsafe {
*(remainder as *mut u64) = dividend;
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


#[inline(always)]
fn mul_u64_u32_add_u64_shr(a: u64, mul: u32, b: u64, shift: u32) -> u64 {
((a as u128).wrapping_mul(mul as u128).wrapping_add(b as u128) >> shift) as u64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
}


fn div_u64_rem(dividend: u64, divisor: u32, remainder: &mut u32) -> u64 {
unsafe {
*(remainder as *mut u32) = (dividend % (divisor as u64)) as u32;
}
dividend / (divisor as u64)
}

fn div64_u64_rem(dividend: u64, divisor: u64, remainder: &mut u64) -> u64 {
unsafe {
*(remainder as *mut u64) = dividend % divisor;
}
dividend / divisor
}

fn div64_u64(dividend: u64, divisor: u64) -> u64 {
dividend / divisor
}

fn div_u64(dividend: u64, divisor: u32) -> u64 {
let mut remainder: u32 = 0;
div_u64_rem(dividend, divisor, unsafe { &mut (*core::ptr::addr_of_mut!(remainder)) })
}

// DIV_U64_ROUND_UP - unsigned 64bit divide with 32bit divisor rounded up
// @ll: unsigned 64bit dividend
// @d: unsigned 32bit divisor
//
// Divide unsigned 64bit dividend by unsigned 32bit divisor
// and round up.
//
// Return: dividend / divisor rounded up
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
struct timezone {
    tz_minuteswest: i32,
    tz_dsttime: i32,
}


// The IDs of the various system clocks (for POSIX.1b interval timers):
// AUX clock support. AUXiliary clocks are dynamically configured by
// enabling a clock ID. These clock can be steered independently of the
// core timekeeper. The kernel can support up to 8 auxiliary clocks, but
// the actual limit depends on eventual architecture constraints vs. VDSO.
#[repr(C)]
#[derive(Clone, Copy)]
struct timespec64 {
    tv_sec: i64,
    tv_nsec: i64,
}


fn timespec64_compare(lhs: *const timespec64, rhs: *const timespec64) -> i32 {
if (unsafe { (*lhs).tv_sec }) < unsafe { (*rhs).tv_sec } {
return 1i32.wrapping_neg();
}
if (unsafe { (*lhs).tv_sec }) > unsafe { (*rhs).tv_sec } {
return 1;
}
unsafe { (*lhs).tv_nsec }.wrapping_sub(unsafe { (*rhs).tv_nsec }) as i32
}

fn timespec64_add(mut lhs: timespec64, rhs: timespec64) -> timespec64 {
let mut ts_delta: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ts_delta) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
unsafe { set_normalized_timespec64(core::ptr::addr_of_mut!(ts_delta), lhs.tv_sec.wrapping_add(rhs.tv_sec), lhs.tv_nsec.wrapping_add(rhs.tv_nsec)) };
return ts_delta;
}

fn timespec64_sub(mut lhs: timespec64, rhs: timespec64) -> timespec64 {
let mut ts_delta: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ts_delta) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
unsafe { set_normalized_timespec64(core::ptr::addr_of_mut!(ts_delta), lhs.tv_sec.wrapping_sub(rhs.tv_sec), lhs.tv_nsec.wrapping_sub(rhs.tv_nsec)) };
return ts_delta;
}

fn timespec64_valid(ts: *const timespec64) -> bool {
// Dates before 1970 are bogus
if (unsafe { (*ts).tv_sec }) < 0 {
return false;
}
// Can't have more nanoseconds then a second
if ((unsafe { (*ts).tv_nsec }) as u64) >= 1000000000 {
return false;
}
true
}

fn timespec64_valid_strict(ts: *const timespec64) -> bool {
if !timespec64_valid(ts) {
return false;
}
// Disallow values that could overflow ktime_t
if ((unsafe { (*ts).tv_sec }) as u64) >= 9223372036u64 {
return false;
}
true
}

fn timespec64_valid_settod(ts: *const timespec64) -> bool {
if !timespec64_valid(ts) {
return false;
}
// Disallow values which cause overflow issues vs. CLOCK_REALTIME
if ((unsafe { (*ts).tv_sec }) as u64) >= 8277292036u64 {
return false;
}
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


fn timespec64_to_ns(ts: &timespec64) -> i64 {
// Prevent multiplication overflow / underflow
if (unsafe { (*(ts as *const timespec64)).tv_sec }) >= 9223372036i64 {
return 9223372036854775807i64;
}
if (unsafe { (*(ts as *const timespec64)).tv_sec }) <= 9223372036854775807i64.wrapping_neg().wrapping_sub(1) / 1000000000 {
return 9223372036854775807i64.wrapping_neg().wrapping_sub(1);
}
unsafe { (*(ts as *const timespec64)).tv_sec }.wrapping_mul(1000000000).wrapping_add(unsafe { (*(ts as *const timespec64)).tv_nsec })
}

#[inline(always)]
fn timespec64_add_ns(a: *mut timespec64, mut ns: u64) {
let __v0: u32 = __iter_div_u64_rem(((unsafe { (*a).tv_nsec }) as u64).wrapping_add(ns), 1000000000u32, unsafe { &mut (*core::ptr::addr_of_mut!(ns)) });
let __v2: i64 = unsafe { (*a).tv_sec };
unsafe {
(*a).tv_sec = __v2.wrapping_add(__v0 as i64);
}
unsafe {
(*a).tv_nsec = ns as i64;
}
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
// Thread-synchronous status.
//
// This is different from the flags in that nobody else
// ever touches our thread-synchronous status, so we don't
// have to worry about atomic accesses.
// 32bit syscall active (64BIT)
// regs poked by 32-bit ptracer
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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
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
struct workqueue_struct {
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
struct rseq {
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
struct rb_root {
    rb_node: *mut rb_node,
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
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rb_root_linked {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node_linked,
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
struct timerqueue_linked_head {
    rb_root: rb_root_linked,
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
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
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
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
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
struct k_sigaction {
    sa: sigaction,
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
struct __call_single_data {
    node: __call_single_node,
    func: Option<unsafe extern "C" fn(*mut ())>,
    info: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seq_file {
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
static class_raw_spinlock_irqsave_is_conditional: bool = 0i32 != 0i32;

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_594 {
    lock: *mut raw_spinlock,
    flags: u64,
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
fn class_raw_spinlock_irqsave_destructor(_T: *mut __SlateRecord_spinlock_h_594) {
raw_spin_unlock_irqrestore!(unsafe { (*_T).lock }, unsafe { (*_T).flags });
}

#[inline(always)]
fn class_raw_spinlock_irqsave_lock_ptr(_T: &mut __SlateRecord_spinlock_h_594) -> *mut () {
let mut _ptr: *mut () = ((unsafe { *core::ptr::addr_of_mut!((*(_T as *mut __SlateRecord_spinlock_h_594)).lock) }) as u64) as *mut ();
if IS_ERR(_ptr as *const ()) {
_ptr = core::ptr::null_mut::<()>();
}
_ptr
}

#[inline(always)]
fn class_raw_spinlock_irqsave_constructor(l: *mut raw_spinlock) -> __SlateRecord_spinlock_h_594 {
let mut _t: __SlateRecord_spinlock_h_594 = {
let mut __t0: __SlateRecord_spinlock_h_594 = unsafe { core::mem::zeroed() };
__t0.lock = l;
__t0
};
let _T: *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(_t);
raw_spin_lock_irqsave__2!(unsafe { (*_T).lock }, (*_T).flags);
return _t;
}

#[inline(always)]
fn __class_raw_spinlock_irqsave_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_594) {
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct rseq_event {
    __slate_anon_0: __SlateRecord_rseq_types_h_38,
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
struct rseq_ids {
    __slate_anon_0: __SlateRecord_rseq_types_h_74,
    node_id: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct vm_struct {
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
struct kmem_cache {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_raw_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_spinlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_rwlock {
    seqcount: seqcount,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct seqcount_mutex {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits288U0,
    dl_timer: hrtimer,
    inactive_timer: hrtimer,
    rq: *mut rq,
    server_pick_task: Option<unsafe extern "C" fn(*mut sched_dl_entity, *mut rq_flags) -> *mut task_struct>,
    pi_se: *mut sched_dl_entity,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
union rcu_special {
    b: __SlateRecord_sched_h_791,
    s: u32,
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
    __slate_bits_0: __slate_bits::__SlateBits66U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits66U1,
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
    __slate_bits_2: __slate_bits::__SlateBits66U2,
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
struct xol_area {
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
#[repr(C)]
#[derive(Clone, Copy)]
struct ldt_struct {
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
struct mem_cgroup {
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
struct completion {
    done: u32,
    wait: swait_queue_head,
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
struct __kernel_timex_timeval {
    tv_sec: i64,
    tv_usec: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __kernel_timex {
    modes: u32,
    __slate_bits_0: __slate_bits::__SlateBits336U0,
    offset: i64,
    freq: i64,
    maxerror: i64,
    esterror: i64,
    status: i32,
    __slate_bits_1: __slate_bits::__SlateBits336U1,
    constant: i64,
    precision: i64,
    tolerance: i64,
    time: __kernel_timex_timeval,
    tick: i64,
    ppsfreq: i64,
    jitter: i64,
    shift: i32,
    __slate_bits_2: __slate_bits::__SlateBits336U2,
    stabil: i64,
    jitcnt: i64,
    calcnt: i64,
    errcnt: i64,
    stbcnt: i64,
    tai: i32,
    __slate_bits_3: __slate_bits::__SlateBits336U3,
}


// frequency offset
// clock status
// set TAI offset
// add 'time' to current time
// select nanosecond resolution
// tick value
// enable PPS freq discipline (rw)
// enable PPS time discipline (rw)
// insert leap (rw)
// delete leap (rw)
// old-fashioned adjtime
// read-only adjtime
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.random_get_entropy_fallback")]
extern "C" fn random_get_entropy_fallback() -> u64 {
let tkr: *mut tk_read_base = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper.tkr_mono) };
let mut __v0: *mut clocksource;
loop {
if !(__native_word__57!() || ((const { core::mem::size_of::<*mut clocksource>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_605() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*tkr).clock) as *const *mut clocksource) };
let clock: *mut clocksource = __v0;
if unlikely__2!((unsafe { timekeeping_suspended }) != 0 || clock.is_null()) != 0 {
return 0;
}
unsafe { unsafe { (*clock).read }.unwrap()(clock) }
}

#[inline(always)]
fn rdtsc_ordered() -> u64 {
let mut low: u64 = 0;
let mut high: u64 = 0;
// The RDTSC instruction is not ordered relative to memory
// access.  The Intel SDM and the AMD APM are both vague on this
// point, but empirically an RDTSC instruction can be
// speculatively executed before prior loads.  An RDTSC
// immediately after an appropriate barrier appears to be
// ordered as a normal load, that is, it provides the same
// ordering guarantees as reading from a global memory location
// that some other imaginary CPU is updating continuously with a
// time stamp.
//
// Thus, use the preferred barrier on the respective CPU, aiming for
// RDTSCP as the default.
unsafe {
let __t0: u64;
let __t1: u64;
core::arch::asm!("# ALT: oldinstr\n771:\n\t# ALT: oldinstr\n771:\n\trdtsc\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte (20*32+ 2)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\tlfence; rdtsc\n775:\n.popsection\n\n772:\n# ALT: padding\n.skip -(((775f-774f)-(772b-771b)) > 0) * ((775f-774f)-(772b-771b)),0x90\n773:\n.pushsection .altinstructions, \"aM\", @progbits, 14\n .long 771b - .\n .long 774f - .\n .4byte ( 1*32+27)\n .byte 773b-771b\n .byte 775f-774f\n.popsection\n.pushsection .altinstr_replacement, \"ax\"\n912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n# ALT: replacement\n774:\n\trdtscp\n775:\n.popsection\n", lateout("ax") __t0, lateout("dx") __t1, out("cx") _, options(nostack, att_syntax));
low = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
high = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
// RDTSCP clobbers ECX with MSR_TSC_AUX.
EAX_EDX_VAL!(low, high)
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
struct old_timespec32 {
    tv_sec: i32,
    tv_nsec: i32,
}


fn ktime_set(secs: i64, nsecs: u64) -> i64 {
if unlikely__2!(secs >= 9223372036i64) != 0 {
return 9223372036854775807i64;
}
secs.wrapping_mul(1000000000).wrapping_add(nsecs as i64)
}

fn timespec64_to_ktime(ts: timespec64) -> i64 {
ktime_set(ts.tv_sec, ts.tv_nsec as u64)
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


fn ktime_to_ns(kt: i64) -> i64 {
kt
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


fn ktime_compare(cmp1: i64, cmp2: i64) -> i32 {
if cmp1 < cmp2 {
return 1i32.wrapping_neg();
}
if cmp1 > cmp2 {
return 1;
}
0
}

fn ktime_after(cmp1: i64, cmp2: i64) -> bool {
ktime_compare(cmp1, cmp2) > 0
}

fn ns_to_ktime(ns: u64) -> i64 {
ns as i64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..read_mostly")]
static mut timekeeping_suspended: i32 = unsafe { core::mem::zeroed() };

#[inline(never)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".noinstr.text")]
extern "C" fn __ktime_get_real_seconds() -> i64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_565() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*tk).xtime_sec) as *const u64) };
__v0 as i64
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mnt_idmap {
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_real_seconds")]
extern "C" fn ktime_get_real_seconds() -> i64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seconds: i64 = 0;
let mut seq: u32 = 0;
if IS_ENABLED__2!() != 0 {
let mut __v0: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_562() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*tk).xtime_sec) as *const u64) };
return __v0 as i64;
}
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
let mut __v7: u64;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_563() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v7 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*tk).xtime_sec) as *const u64) };
seconds = __v7 as i64;
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
seconds
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_mono_to_any")]
extern "C" fn ktime_mono_to_any(tmono: i64, offs: u32) -> i64 {
let offset: *const i64 = unsafe { offsets.0[(offs as u32) as usize] };
let mut seq: u32 = 0;
let mut tconv: i64 = 0;
if IS_ENABLED__2!() != 0 {
// Paired with WRITE_ONCE()s in tk_set_wall_to_mono() and
// tk_update_sleep_time().
let mut __v0: i64;
loop {
if !(__native_word__14!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_553() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(offset as *const i64) };
return tmono.wrapping_add(__v0);
}
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
tconv = ktime_add!(tmono, unsafe { *offset });
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
tconv
}

#[repr(C)]
#[derive(Clone, Copy)]
struct system_time_snapshot {
    cycles: u64,
    hw_cycles: u64,
    systime: i64,
    monoraw: i64,
    cs_id: u32,
    hw_csid: u32,
    clock_was_set_seq: u32,
    cs_was_changed_seq: u8,
    valid: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct system_counterval_t {
    cycles: u64,
    cs_id: u32,
    use_nsecs: bool,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct system_device_crosststamp {
    clock_id: i32,
    device: i64,
    sys_counter: system_counterval_t,
    sys_systime: i64,
    sys_monoraw: i64,
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


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.timekeeping_clocksource_has_base")]
extern "C" fn timekeeping_clocksource_has_base(id: u32) -> bool {
let mut __v0: *mut clocksource_base;
loop {
if !(__native_word__44!() || ((const { core::mem::size_of::<*mut clocksource_base>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_578() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*timekeeper_data[0].timekeeper.tkr_mono.clock).base) as *const *mut clocksource_base) };
// This is a snapshot, so no point in using the sequence
// count. Just prevent the compiler from re-evaluating @base as the
// clocksource might change concurrently.
let base: *mut clocksource_base = __v0;
!base.is_null() && ((unsafe { (*base).id }) as u32) == (id as u32)
}

#[unsafe(no_mangle)]
static mut persistent_clock_is_local: i32 = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct assoc_array_ptr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct net {
}


fn __seqcount_init(s: *mut seqcount, name: *const u8, key: *mut lock_class_key) {
// Make sure we are not reinitializing a held lock:
lockdep_init_map!(name, key);
unsafe {
(*s).sequence = 0;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct key_type {
}


// SEQCNT_ZERO() - static initializer for seqcount_t
// @name: Name of the seqcount_t instance
// SEQCOUNT_LOCKNAME()	- Instantiate seqcount_LOCKNAME_t and helpers
// seqprop_LOCKNAME_*()	- Property accessors for seqcount_LOCKNAME_t
//
// @lockname:		"LOCKNAME" part of seqcount_LOCKNAME_t
// @locktype:		LOCKNAME canonical C data type
// @preemptible:	preemptibility of above locktype
// @lockbase:		prefix for associated lock/unlock
// 									 * Re-read the sequence counter since the (possibly			 * preempted) writer made progress.
// PREEMPT_RT relies on the above LOCK+UNLOCK
fn __seqprop_ptr(s: &seqcount) -> *mut seqcount {
(s as *const seqcount) as *mut seqcount
}

fn __seqprop_const_ptr(s: &seqcount) -> *const seqcount {
s as *const seqcount
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


fn __seqprop_sequence(s: &seqcount) -> u32 {
let mut __v0: u32;
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_195() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount)).sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_196() };
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
__v0
}

fn __seqprop_preemptible(s: &seqcount) -> bool {
false
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


fn __seqprop_assert(s: &seqcount) {
lockdep_assert_preemption_disabled!();
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[inline(always)]
fn __seqprop_raw_spinlock_ptr(s: *mut seqcount_raw_spinlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*s).seqcount) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[inline(always)]
fn __seqprop_raw_spinlock_const_ptr(s: *const seqcount_raw_spinlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*s).seqcount) }
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
fn __seqprop_raw_spinlock_sequence(s: *const seqcount_raw_spinlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_197() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_198() };
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
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_199() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_200() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
}

#[inline(always)]
fn __seqprop_raw_spinlock_preemptible(s: *const seqcount_raw_spinlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
}

#[inline(always)]
fn __seqprop_raw_spinlock_assert(s: *const seqcount_raw_spinlock) {
}

#[inline(always)]
fn __seqprop_spinlock_ptr(s: &mut seqcount_spinlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_spinlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_spinlock_const_ptr(s: &seqcount_spinlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_spinlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_spinlock_sequence(s: &seqcount_spinlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_201() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_spinlock)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_202() };
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
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_203() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_spinlock)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_204() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
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


#[inline(always)]
fn __seqprop_spinlock_preemptible(s: &seqcount_spinlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
}

#[inline(always)]
fn __seqprop_spinlock_assert(s: &seqcount_spinlock) {
}

#[inline(always)]
fn __seqprop_rwlock_ptr(s: &mut seqcount_rwlock) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_rwlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_rwlock_const_ptr(s: &seqcount_rwlock) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount) }
}

#[inline(always)]
fn __seqprop_rwlock_sequence(s: &seqcount_rwlock) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_205() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_206() };
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
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if false && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_207() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_rwlock)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_208() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
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


#[inline(always)]
fn __seqprop_rwlock_preemptible(s: &seqcount_rwlock) -> bool {
if IS_ENABLED!() == 0 {
return false;
}
false
}

#[inline(always)]
fn __seqprop_rwlock_assert(s: &seqcount_rwlock) {
}

#[inline(always)]
fn __seqprop_mutex_ptr(s: &mut seqcount_mutex) -> *mut seqcount {
unsafe { core::ptr::addr_of_mut!((*(s as *mut seqcount_mutex)).seqcount) }
}

#[inline(always)]
fn __seqprop_mutex_const_ptr(s: &seqcount_mutex) -> *const seqcount {
unsafe { core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount) }
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


#[inline(always)]
fn __seqprop_mutex_sequence(s: &seqcount_mutex) -> u32 {
let mut seq: u32 = 0;
let mut __v0: u32;
{
let mut ___p1: u32 = 0;
let mut __v1: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_209() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount.sequence) as *const u32) };
___p1 = __v1;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_210() };
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
seq = __v0;
if IS_ENABLED!() == 0 {
return seq;
}
if true && unlikely__3!(seq & 1) != 0 {
let mut __v4: u32;
let mut ___p1: u32 = 0;
let mut __v5: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_211() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v5 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*(s as *const seqcount_mutex)).seqcount.sequence) as *const u32) };
___p1 = __v5;
loop {
if !__native_word__3!() {
unsafe { __compiletime_assert_212() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
__v4 = ___p1;
seq = __v4;
}
seq
}

#[inline(always)]
fn __seqprop_mutex_preemptible(s: &seqcount_mutex) -> bool {
if IS_ENABLED!() == 0 {
return true;
}
false
}

#[inline(always)]
fn __seqprop_mutex_assert(s: &seqcount_mutex) {
}

// __read_seqcount_begin() - begin a seqcount_t read section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Return: count to be passed to read_seqcount_retry()
// raw_read_seqcount_begin() - begin a seqcount_t read section w/o lockdep
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
fn do___read_seqcount_retry(s: *const seqcount, start: u32) -> i32 {
kcsan_atomic_next(0i32);
let mut __v0: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_213() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).sequence) as *const u32) };
((__v0 != start) as i64) as i32
}

fn do_read_seqcount_retry(s: *const seqcount, start: u32) -> i32 {
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
do___read_seqcount_retry(s, start)
}

fn do_raw_write_seqcount_begin(__v0: *mut seqcount) {
kcsan_nestable_atomic_begin();
let __v1: u32 = unsafe { (*__v0).sequence };
unsafe {
(*__v0).sequence = __v1.wrapping_add(1);
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
}

fn do_raw_write_seqcount_end(__v2: *mut seqcount) {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
let __v3: u32 = unsafe { (*__v2).sequence };
unsafe {
(*__v2).sequence = __v3.wrapping_add(1);
}
kcsan_nestable_atomic_end();
}

fn do_write_seqcount_begin_nested(s: *mut seqcount, subclass: i32) {
seqcount_acquire!();
do_raw_write_seqcount_begin(s);
}

// write_seqcount_begin() - start a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: sequence counter write side sections must be serialized and
// non-preemptible. Preemption will be automatically disabled if and
// only if the seqcount write serialization lock is associated, and
// preemptible.  If readers can be invoked from hardirq or softirq
// context, interrupts or bottom halves must be respectively disabled.
fn do_write_seqcount_begin(s: *mut seqcount) {
do_write_seqcount_begin_nested(s, 0i32);
}

// write_seqcount_end() - end a seqcount_t write side critical section
// @s: Pointer to seqcount_t or any of the seqcount_LOCKNAME_t variants
//
// Context: Preemption will be automatically re-enabled if and only if
// the seqcount write serialization lock is associated, and preemptible.
fn do_write_seqcount_end(s: *mut seqcount) {
seqcount_release!();
do_raw_write_seqcount_end(s);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_seqlock_h_612 {
    seqcount: seqcount,
}


// SEQCNT_LATCH_ZERO() - static initializer for seqcount_latch_t
// @seq_name: Name of the seqcount_latch_t instance
#[inline(always)]
fn raw_read_seqcount_latch(s: *const __SlateRecord_seqlock_h_612) -> u32 {
// Pairs with the first smp_wmb() in raw_write_seqcount_latch().
// Due to the dependent load, a full smp_rmb() is not needed.
let mut __v0: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_214() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).seqcount.sequence) as *const u32) };
__v0
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
}


#[inline(always)]
fn read_seqcount_latch(s: *const __SlateRecord_seqlock_h_612) -> u32 {
kcsan_atomic_next(1000i32);
raw_read_seqcount_latch(s)
}

#[inline(always)]
fn raw_read_seqcount_latch_retry(s: *const __SlateRecord_seqlock_h_612, start: u32) -> i32 {
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
let mut __v4: u32;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_215() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v4 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*s).seqcount.sequence) as *const u32) };
((__v4 != start) as i64) as i32
}

#[inline(always)]
fn read_seqcount_latch_retry(s: *const __SlateRecord_seqlock_h_612, start: u32) -> i32 {
kcsan_atomic_next(0i32);
raw_read_seqcount_latch_retry(s, start)
}

#[inline(always)]
fn raw_write_seqcount_latch(__v2: *mut __SlateRecord_seqlock_h_612) {
loop {
kcsan_wmb!();
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
if true {
break;
}
}
// prior stores before incrementing "sequence"
let __v3: u32 = unsafe { (*__v2).seqcount.sequence };
unsafe {
(*__v2).seqcount.sequence = __v3.wrapping_add(1);
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
// increment "sequence" before following stores
}

#[inline(always)]
fn write_seqcount_latch_begin(s: *mut __SlateRecord_seqlock_h_612) {
kcsan_nestable_atomic_begin();
raw_write_seqcount_latch(s);
}

#[inline(always)]
fn write_seqcount_latch(s: *mut __SlateRecord_seqlock_h_612) {
raw_write_seqcount_latch(s);
}

#[inline(always)]
fn write_seqcount_latch_end(s: *mut __SlateRecord_seqlock_h_612) {
kcsan_nestable_atomic_end();
}

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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
struct srcu_struct {
    srcu_ctrp: *mut srcu_ctr,
    sda: *mut srcu_data,
    srcu_reader_flavor: u8,
    dep_map: lockdep_map,
    srcu_sup: *mut srcu_usage,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct notifier_block {
    notifier_call: Option<unsafe extern "C" fn(*mut notifier_block, u64, *mut ()) -> i32>,
    next: *mut notifier_block,
    priority: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct raw_notifier_head {
    head: *mut notifier_block,
}


// Suits me
// Don't call further
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
struct hprobe {
    state: u32,
    srcu_scp: *mut srcu_ctr,
    uprobe: *mut uprobe,
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
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_mm_types_h_41 {
    f: u64,
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
struct __SlateRecord_mm_types_h_288 {
    val: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
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
struct mmap_action {
    __slate_anon_0: __SlateRecord_mm_types_h_830,
    r#type: u32,
    error_override: i32,
    __slate_bits_0: __slate_bits::__SlateBits419U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
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
struct kernfs_open_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_iattrs {
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
struct kernfs_root {
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
struct ctl_table_poll {
    event: __SlateRecord_types_h_190,
    wait: wait_queue_head,
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
struct sock {
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


#[repr(C)]
#[derive(Clone, Copy)]
struct cgroup_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct time_namespace {
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
    __slate_bits_0: __slate_bits::__SlateBits291U0,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct trace_eval_map {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits501U0,
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
union __SlateRecord_timekeeper_internal_h_156 {
    offs_tai: i64,
    offs_aux: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_timekeeper_internal_h_176 {
    monotonic_to_boot: timespec64,
    monotonic_to_aux: timespec64,
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
struct __SlateRecord_timekeeping_c_141 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timekeeping_c_142 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timekeeping_c_147 {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_timekeeping_c_148 {
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


#[repr(C, align(16))]
struct __SlateAlign16<T>(T);

unsafe impl<T> Sync for __SlateAlign16<T> {
}

#[repr(C, align(64))]
struct __SlateAlign64<T>(T);

unsafe impl<T> Sync for __SlateAlign64<T> {
}

#[repr(transparent)]
struct __SlateConst<T>(T);

unsafe impl<T> Sync for __SlateConst<T> {
}

mod __slate_bits {
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits288U0 {
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
    pub struct __SlateBits501U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits640U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits591U0 {
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
    pub struct __SlateBits419U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits66U0 {
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
    pub struct __SlateBits66U1 {
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
    pub struct __SlateBits291U0 {
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
    pub struct __SlateBits66U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits113U0 {
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
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits336U0 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits336U1 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits336U2 {
        #[bits(32, access = na)]
        pub __slate_pad_0: u32,
    }
    #[bitfields::bitfield([u8; 44], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits336U3 {
        #[bits(128, access = na)]
        pub __slate_pad_0: u128,
        #[bits(128, access = na)]
        pub __slate_pad_1: u128,
        #[bits(96, access = na)]
        pub __slate_pad_2: u128,
    }
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_mono_fast_ns: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_raw_fast_ns: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_boot_fast_ns: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_tai_fast_ns: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_real_fast_ns: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pvclock_gtod_register_notifier: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_pvclock_gtod_unregister_notifier: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_real_ts64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_resolution_ns: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_with_offset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_coarse_with_offset: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_mono_to_any: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_raw: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_ts64: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_seconds: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_real_seconds: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_snapshot_id: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_real_to_base_clock: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_get_device_system_crosststamp: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_timekeeping_clocksource_has_base: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_do_settimeofday64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_raw_ts64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_timekeeping__590_2343_timekeeping_init_ops6:\t\t\t\n.long\t{23} - .\t\n.previous\t\t\t\t\t\n\n.section \".export_symbol\",\"a\" ; __export_symbol_getboottime64: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_coarse_real_ts64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_ktime_get_coarse_ts64: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_random_get_entropy_fallback: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous", sym ktime_get_mono_fast_ns, sym ktime_get_raw_fast_ns, sym ktime_get_boot_fast_ns, sym ktime_get_tai_fast_ns, sym ktime_get_real_fast_ns, sym pvclock_gtod_register_notifier, sym pvclock_gtod_unregister_notifier, sym ktime_get_real_ts64, sym ktime_get, sym ktime_get_resolution_ns, sym ktime_get_with_offset, sym ktime_get_coarse_with_offset, sym ktime_mono_to_any, sym ktime_get_raw, sym ktime_get_ts64, sym ktime_get_seconds, sym ktime_get_real_seconds, sym ktime_get_snapshot_id, sym ktime_real_to_base_clock, sym get_device_system_crosststamp, sym timekeeping_clocksource_has_base, sym do_settimeofday64, sym ktime_get_raw_ts64, sym timekeeping_init_ops, sym getboottime64, sym ktime_get_coarse_real_ts64, sym ktime_get_coarse_ts64, sym random_get_entropy_fallback, options(att_syntax));

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
    __slate_bits_0: __slate_bits::__SlateBits640U0,
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
    __slate_bits_0: __slate_bits::__SlateBits591U0,
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
struct audit_ntp_val {
    oldval: i64,
    newval: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct audit_ntp_data {
    vals: [audit_ntp_val; 6],
}


// maximized args number that audit_socketcall can process
// dentry represents the parent
// audit record should be hidden
fn audit_context() -> *mut audit_context {
unsafe { (*get_current()).audit_context }
}

fn audit_dummy_context() -> bool {
let p: *mut () = audit_context() as *mut ();
p.is_null() || (unsafe { *(p as *mut i32) }) != 0
}

fn audit_tk_injoffset(offset: timespec64) {
// ignore no-op events
if offset.tv_sec == 0 && offset.tv_nsec == 0 {
return;
}
if !audit_dummy_context() {
unsafe { __audit_tk_injoffset(offset) };
}
}

fn audit_ntp_log(ad: *const audit_ntp_data) {
if !audit_dummy_context() {
unsafe { __audit_ntp_log(ad) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct clocksource_hw_snapshot {
    hw_cycles: u64,
    hw_csid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct clocksource {
    read: Option<unsafe extern "C" fn(*mut clocksource) -> u64>,
    mask: u64,
    mult: u32,
    shift: u32,
    max_idle_ns: u64,
    maxadj: u32,
    max_cycles: u64,
    max_raw_delta: u64,
    name: *const u8,
    list: list_head,
    freq_khz: u32,
    rating: i32,
    id: u32,
    vdso_clock_mode: u32,
    flags: u64,
    base: *mut clocksource_base,
    read_snapshot: Option<unsafe extern "C" fn(*mut clocksource, *mut clocksource_hw_snapshot) -> u64>,
    enable: Option<unsafe extern "C" fn(*mut clocksource) -> i32>,
    disable: Option<unsafe extern "C" fn(*mut clocksource)>,
    suspend: Option<unsafe extern "C" fn(*mut clocksource)>,
    resume: Option<unsafe extern "C" fn(*mut clocksource)>,
    mark_unstable: Option<unsafe extern "C" fn(*mut clocksource)>,
    tick_stable: Option<unsafe extern "C" fn(*mut clocksource)>,
    wd_list: list_head,
    cs_last: u64,
    wd_last: u64,
    wd_cpu: u32,
    owner: *mut module,
}


// simplify initialization of mask field
#[repr(C)]
#[derive(Clone, Copy)]
struct clocksource_base {
    id: u32,
    freq_khz: u32,
    offset: u64,
    numerator: u32,
    denominator: u32,
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


fn touch_softlockup_watchdog() {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct syscore_ops {
    suspend: Option<unsafe extern "C" fn(*mut ()) -> i32>,
    resume: Option<unsafe extern "C" fn(*mut ())>,
    shutdown: Option<unsafe extern "C" fn(*mut ())>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct syscore {
    node: list_head,
    ops: *const syscore_ops,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tk_read_base {
    clock: *mut clocksource,
    mask: u64,
    cycle_last: u64,
    mult: u32,
    shift: u32,
    xtime_nsec: u64,
    base: i64,
    base_real: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct timekeeper {
    tkr_mono: tk_read_base,
    xtime_sec: u64,
    ktime_sec: u64,
    wall_to_monotonic: timespec64,
    offs_real: i64,
    offs_boot: i64,
    __slate_anon_6: __SlateRecord_timekeeper_internal_h_156,
    coarse_nsec: u32,
    id: u32,
    tkr_raw: tk_read_base,
    raw_sec: u64,
    cs_id: u32,
    cs_ns_to_cyc_mult: u32,
    cs_ns_to_cyc_shift: u32,
    cs_ns_to_cyc_maxns: u64,
    clock_was_set_seq: u32,
    cs_was_changed_seq: u8,
    clock_valid: u8,
    __slate_anon_18: __SlateRecord_timekeeper_internal_h_176,
    cycle_interval: u64,
    xtime_interval: u64,
    raw_interval: u64,
    next_leap_ktime: i64,
    ntp_tick: u64,
    ntp_error: i64,
    ntp_error_shift: u32,
    ntp_err_mult: u32,
    cs_tick_adj: i64,
    skip_second_overflow: u32,
    skew_delta: i64,
    tai_offset: i32,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
    cpu_base: *mut hrtimer_cpu_base,
    index: u32,
    clockid: i32,
    seq: seqcount_raw_spinlock,
    expires_next: i64,
    running: *mut hrtimer,
    active: timerqueue_linked_head,
    offset: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_cpu_base {
    lock: raw_spinlock,
    cpu: u32,
    active_bases: u32,
    clock_was_set_seq: u32,
    hres_active: bool,
    deferred_rearm: bool,
    deferred_needs_update: bool,
    hang_detected: bool,
    softirq_activated: bool,
    online: bool,
    nr_events: u32,
    nr_retries: u16,
    nr_hangs: u16,
    max_hang_time: u32,
    expires_next: i64,
    next_timer: *mut hrtimer,
    softirq_expires_next: i64,
    softirq_next_timer: *mut hrtimer,
    deferred_expires_next: i64,
    clock_base: [hrtimer_clock_base; 8],
    csd: __call_single_data,
}


#[unsafe(no_mangle)]
extern "C" fn ktime_expiry_to_cycles(id: u32, expires_ns: i64, cycles: *mut u64) -> bool {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let tkrm: *mut tk_read_base = unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) };
let mut base_ns: i64 = 0;
let mut delta_ns: i64 = 0;
let mut max_ns: i64 = 0;
let mut base_cycles: u64 = 0;
let mut delta_cycles: u64 = 0;
let mut seq: u32 = 0;
let mut mult: u32 = 0;
let mut shift: u32 = 0;
// Racy check to avoid the seqcount overhead when ID does not match. If
// the relevant clocksource is installed concurrently, then this will
// just delay the switch over to this mechanism until the next event is
// programmed. If the ID is not matching the clock events code will use
// the regular relative set_next_event() callback as before.
if (data_race__2!(unsafe { (*tk).cs_id }) as u32) != (id as u32) {
return false;
}
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
if ((unsafe { (*tk).cs_id }) as u32) != (id as u32) {
return false;
}
base_cycles = unsafe { (*tkrm).cycle_last };
base_ns = ((unsafe { (*tkrm).base }) as u64).wrapping_add((unsafe { (*tkrm).xtime_nsec }) >> unsafe { (*tkrm).shift }) as i64;
mult = unsafe { (*tk).cs_ns_to_cyc_mult };
shift = unsafe { (*tk).cs_ns_to_cyc_shift };
max_ns = (unsafe { (*tk).cs_ns_to_cyc_maxns }) as i64;
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
// Prevent negative deltas and multiplication overflows
let mut __v8: i64;
let __UNIQUE_ID_x__532: i64 = expires_ns.wrapping_sub(base_ns);
let __UNIQUE_ID_y__533: i64 = max_ns;
loop {
if __types_ok__2!(__UNIQUE_ID_x__532, __UNIQUE_ID_y__533) == 0 {
unsafe { __compiletime_assert_534() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v8 = if __UNIQUE_ID_x__532 < __UNIQUE_ID_y__533 { __UNIQUE_ID_x__532 } else { __UNIQUE_ID_y__533 };
delta_ns = __v8;
let mut __v10: i64;
let __UNIQUE_ID_x__535: i64 = delta_ns;
let __UNIQUE_ID_y__536: i32 = 0;
loop {
if __types_ok__7!(__UNIQUE_ID_x__535, __UNIQUE_ID_y__536) == 0 {
unsafe { __compiletime_assert_537() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = if __UNIQUE_ID_x__535 > (__UNIQUE_ID_y__536 as i64) { __UNIQUE_ID_x__535 } else { __UNIQUE_ID_y__536 as i64 };
delta_ns = __v10;
// Convert to cycles
delta_cycles = (delta_ns as u64).wrapping_mul(mult as u64) >> shift;
unsafe {
*cycles = base_cycles.wrapping_add(delta_cycles);
}
true
}

fn timekeeping_inc_mg_floor_swaps() {
loop {
__verify_pcpu_ptr__6!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if true && (true || 1 == 1i32.wrapping_neg()) { 1 } else { 0 };
if false {
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = 1u8;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = 1u16;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = 1u32;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_tmp__: u64 = 1;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = 1;
if false {
let pto_tmp__: u64 = 1;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:{0}(%rip)", sym timekeeping_mg_floor_swaps, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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

fn clocksource_delta(now: u64, last: u64, mask: u64, max_delta: u64) -> u64 {
let ret: u64 = now.wrapping_sub(last) & mask;
// Prevent time going backwards by checking the result against
// @max_delta. If greater, return 0.
if ret > max_delta { 0 } else { ret }
}

/// The most important data for readout fits into a single 64 byte
/// cache line.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct tk_data {
    seq: seqcount_raw_spinlock,
    timekeeper: timekeeper,
    shadow_timekeeper: timekeeper,
    lock: raw_spinlock,
}


static mut timekeeper_data: [tk_data; 1] = unsafe { core::mem::zeroed() };

// The core timekeeper
fn tk_get_aux_ts64(tkid: u32, ts: &timespec64) -> bool {
false
}

fn tk_is_aux(tk: *const timekeeper) -> bool {
false
}

fn aux_get_tk_data(id: i32) -> *mut tk_data {
core::ptr::null_mut::<tk_data>()
}

fn tk_update_aux_offs(tk: *mut timekeeper, offs: i64) {
unsafe {
(*tk).__slate_anon_6.offs_aux = offs;
}
unsafe {
(*tk).__slate_anon_18.monotonic_to_aux = ktime_to_timespec64!(offs);
}
}

/// flag for if timekeeping is suspended
/// struct tk_fast - NMI safe timekeeper
/// @seq:	Sequence counter for protecting updates. The lowest bit
/// 		is the index for the tk_read_base array
/// @base:	tk_read_base array. Access is indexed by the lowest bit of
/// 		@seq.
///
/// See @update_fast_timekeeper() below.
#[repr(C)]
#[derive(Clone, Copy)]
struct tk_fast {
    seq: __SlateRecord_seqlock_h_612,
    base: [tk_read_base; 2],
}


/// Suspend-time cycles value for halted fast timekeeper.
static mut cycles_at_suspend: u64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.timekeeping.dummy_clock_read")]
extern "C" fn dummy_clock_read(cs: *mut clocksource) -> u64 {
if (unsafe { timekeeping_suspended }) != 0 {
return unsafe { cycles_at_suspend };
}
unsafe { local_clock() }
}

static mut dummy_clock: clocksource = {
let mut __t0: clocksource = unsafe { core::mem::zeroed() };
__t0.read = Some(dummy_clock_read);
__t0
};

// Boot time initialization which allows local_clock() to be utilized
// during early boot when clocksources are not available. local_clock()
// returns nanoseconds already so no conversion is required, hence mult=1
// and shift=0. When the first proper clocksource is installed then
// the fast time keepers are updated with the correct values.
static mut tk_fast_mono: __SlateAlign64<tk_fast> = __SlateAlign64({
let mut __t0: tk_fast = unsafe { core::mem::zeroed() };
__t0.seq = __SlateRecord_seqlock_h_612 { seqcount: seqcount { sequence: 0u32 } };
__t0.base = [{
let mut __t1: tk_read_base = unsafe { core::mem::zeroed() };
__t1.clock = unsafe { core::ptr::addr_of_mut!(dummy_clock) };
__t1.mask = (((core::mem::size_of::<__SlateRecord_timekeeping_c_141>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(63u64));
__t1.mult = 1u32;
__t1.shift = 0u32;
__t1
}, {
let mut __t2: tk_read_base = unsafe { core::mem::zeroed() };
__t2.clock = unsafe { core::ptr::addr_of_mut!(dummy_clock) };
__t2.mask = (((core::mem::size_of::<__SlateRecord_timekeeping_c_142>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(63u64));
__t2.mult = 1u32;
__t2.shift = 0u32;
__t2
}];
__t0
});

static mut tk_fast_raw: __SlateAlign64<tk_fast> = __SlateAlign64({
let mut __t0: tk_fast = unsafe { core::mem::zeroed() };
__t0.seq = __SlateRecord_seqlock_h_612 { seqcount: seqcount { sequence: 0u32 } };
__t0.base = [{
let mut __t1: tk_read_base = unsafe { core::mem::zeroed() };
__t1.clock = unsafe { core::ptr::addr_of_mut!(dummy_clock) };
__t1.mask = (((core::mem::size_of::<__SlateRecord_timekeeping_c_147>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(63u64));
__t1.mult = 1u32;
__t1.shift = 0u32;
__t1
}, {
let mut __t2: tk_read_base = unsafe { core::mem::zeroed() };
__t2.clock = unsafe { core::ptr::addr_of_mut!(dummy_clock) };
__t2.mask = (((core::mem::size_of::<__SlateRecord_timekeeping_c_148>() as u64) as i32) as u64).wrapping_add((1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) << 0i32 & (1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)).wrapping_sub(1u64).wrapping_add(1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1u64).wrapping_sub((((1i32.wrapping_neg() as u64) < 1u64) as i32) as u64)) >> (core::mem::size_of::<u64>() as u64).wrapping_mul(8u64).wrapping_sub(1u64).wrapping_sub(63u64));
__t2.mult = 1u32;
__t2.shift = 0u32;
__t2
}];
__t0
});

fn tk_aux_setup() {
}

fn tk_aux_update_clocksource() {
}

fn tk_aux_advance() {
}

#[unsafe(no_mangle)]
extern "C" fn timekeeper_lock_irqsave() -> u64 {
let mut flags: u64 = 0;
raw_spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }, flags);
flags
}

#[unsafe(no_mangle)]
extern "C" fn timekeeper_unlock_irqrestore(flags: u64) {
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }, flags);
}

/// Multigrain timestamps require tracking the latest fine-grained timestamp
/// that has been issued, and never returning a coarse-grained timestamp that is
/// earlier than that value.
///
/// mg_floor represents the latest fine-grained time that has been handed out as
/// a file timestamp on the system. This is tracked as a monotonic ktime_t, and
/// converted to a realtime clock value on an as-needed basis.
///
/// Maintaining mg_floor ensures the multigrain interfaces never issue a
/// timestamp earlier than one that has been previously issued.
///
/// The exception to this rule is when there is a backward realtime clock jump. If
/// such an event occurs, a timestamp can appear to be earlier than a previous one.
#[unsafe(link_section = ".data..cacheline_aligned")]
static mut mg_floor: __SlateAlign64<__SlateRecord_types_h_197> = __SlateAlign64(unsafe { core::mem::zeroed() });

fn tk_normalize_xtime(tk: *mut timekeeper) {
while (unsafe { (*tk).tkr_mono.xtime_nsec }) >= 1000000000u64 << unsafe { (*tk).tkr_mono.shift } {
let __v1: u64 = unsafe { (*tk).tkr_mono.xtime_nsec };
unsafe {
(*tk).tkr_mono.xtime_nsec = __v1.wrapping_sub(1000000000u64 << (*tk).tkr_mono.shift);
}
let __v4: u64 = unsafe { (*tk).xtime_sec };
unsafe {
(*tk).xtime_sec = __v4.wrapping_add(1);
}
}
while (unsafe { (*tk).tkr_raw.xtime_nsec }) >= 1000000000u64 << unsafe { (*tk).tkr_raw.shift } {
let __v8: u64 = unsafe { (*tk).tkr_raw.xtime_nsec };
unsafe {
(*tk).tkr_raw.xtime_nsec = __v8.wrapping_sub(1000000000u64 << (*tk).tkr_raw.shift);
}
let __v11: u64 = unsafe { (*tk).raw_sec };
unsafe {
(*tk).raw_sec = __v11.wrapping_add(1);
}
}
}

fn tk_xtime(tk: *const timekeeper) -> timespec64 {
let mut ts: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ts) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
ts.tv_sec = (unsafe { (*tk).xtime_sec }) as i64;
ts.tv_nsec = ((unsafe { (*tk).tkr_mono.xtime_nsec }) >> unsafe { (*tk).tkr_mono.shift }) as i64;
return ts;
}

fn tk_xtime_coarse(tk: *const timekeeper) -> timespec64 {
let mut ts: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ts) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
ts.tv_sec = (unsafe { (*tk).xtime_sec }) as i64;
ts.tv_nsec = (unsafe { (*tk).coarse_nsec }) as i64;
return ts;
}

/// Update the nanoseconds part for the coarse time keepers. They can't rely
/// on xtime_nsec because xtime_nsec could be adjusted by a small negative
/// amount when the multiplication factor of the clock is adjusted, which
/// could cause the coarse clocks to go slightly backwards. See
/// timekeeping_apply_adjustment(). Thus we keep a separate copy for the coarse
/// clockids which only is updated when the clock has been set or  we have
/// accumulated time.
fn tk_update_coarse_nsecs(tk: *mut timekeeper) {
unsafe {
(*tk).coarse_nsec = ((*tk).tkr_mono.xtime_nsec >> (*tk).tkr_mono.shift) as u32;
}
}

fn tk_set_xtime(tk: *mut timekeeper, ts: *const timespec64) {
unsafe {
(*tk).xtime_sec = (*ts).tv_sec as u64;
}
unsafe {
(*tk).tkr_mono.xtime_nsec = ((*ts).tv_nsec as u64) << (*tk).tkr_mono.shift;
}
tk_update_coarse_nsecs(tk);
}

fn tk_xtime_add(tk: *mut timekeeper, ts: *const timespec64) {
let __v1: u64 = unsafe { (*tk).xtime_sec };
unsafe {
(*tk).xtime_sec = __v1.wrapping_add((*ts).tv_sec as u64);
}
let __v4: u64 = unsafe { (*tk).tkr_mono.xtime_nsec };
unsafe {
(*tk).tkr_mono.xtime_nsec = __v4.wrapping_add(((*ts).tv_nsec as u64) << (*tk).tkr_mono.shift);
}
tk_normalize_xtime(tk);
tk_update_coarse_nsecs(tk);
}

fn tk_set_wall_to_mono(tk: *mut timekeeper, mut wtm: timespec64) {
let mut tmp: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
// Verify consistency of: offset_real = -wall_to_monotonic
// before modifying anything
unsafe { set_normalized_timespec64(core::ptr::addr_of_mut!(tmp), unsafe { (*tk).wall_to_monotonic.tv_sec }.wrapping_neg(), unsafe { (*tk).wall_to_monotonic.tv_nsec }.wrapping_neg()) };
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { (*tk).offs_real }) != timespec64_to_ktime(tmp)) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("512: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 512b - ., 3; .popsection\n# {0}", const 512 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_tk_set_wall_to_mono_1, sym __slate_asm_string_tk_set_wall_to_mono_2, const 260 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
(*tk).wall_to_monotonic = wtm;
}
unsafe { set_normalized_timespec64(core::ptr::addr_of_mut!(tmp), wtm.tv_sec.wrapping_neg(), wtm.tv_nsec.wrapping_neg()) };
// Paired with READ_ONCE() in ktime_mono_to_any()
loop {
loop {
if !(__native_word__14!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_514() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__11!((*tk).offs_real, timespec64_to_ktime(tmp));
if true {
break;
}
}
loop {
loop {
if !(__native_word__14!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_515() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__11!((*tk).__slate_anon_6.offs_tai, ktime_add!(unsafe { (*tk).offs_real }, ktime_set((unsafe { (*tk).tai_offset }) as i64, 0u64)));
if true {
break;
}
}
}

fn tk_update_sleep_time(tk: *mut timekeeper, delta: i64) {
// Paired with READ_ONCE() in ktime_mono_to_any()
loop {
loop {
if !(__native_word__14!() || ((const { core::mem::size_of::<i64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_516() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__11!((*tk).offs_boot, ktime_add!(unsafe { (*tk).offs_boot }, delta));
if true {
break;
}
}
// Timespec representation for VDSO update to avoid 64bit division
// on every update.
unsafe {
(*tk).__slate_anon_18.monotonic_to_boot = ktime_to_timespec64!(unsafe { (*tk).offs_boot });
}
}

#[inline(always)]
fn arch_inlined_clocksource_read(cs: *mut clocksource) -> u64 {
rdtsc_ordered()
}

static mut clocksource_read_inlined: static_key_false = static_key_false { key: {
let mut __t0: static_key = unsafe { core::mem::zeroed() };
__t0.enabled = __SlateRecord_types_h_190 { counter: 0i32 };
__t0.__slate_anon_1 = __SlateRecord_jump_label_h_95 { r#type: 0u64 };
__t0
} };

#[inline(always)]
fn tk_clock_read(tkr: *const tk_read_base) -> u64 {
let mut __v0: *mut clocksource;
while false {
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*tkr).clock) as *const *mut clocksource) };
let clock: *mut clocksource = __v0;
if static_branch_likely!(unsafe { core::ptr::addr_of_mut!(clocksource_read_inlined) }) != 0 {
return arch_inlined_clocksource_read(clock);
}
unsafe { unsafe { (*clock).read }.unwrap()(clock) }
}

#[inline(always)]
fn tk_clock_read__const2(tkr: *const tk_read_base) -> u64 {
let mut __v0: *mut clocksource;
while false {
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*tkr).clock) as *const *mut clocksource) };
let clock: *mut clocksource = __v0;
if static_branch_likely!(unsafe { core::ptr::addr_of_mut!(clocksource_read_inlined) }) != 0 {
return arch_inlined_clocksource_read(clock);
}
unsafe { unsafe { (*clock).read }.unwrap()(clock) }
}

#[inline(always)]
fn tk_clock_read__const4(tkr: *const tk_read_base) -> u64 {
let mut __v0: *mut clocksource;
while false {
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*tkr).clock) as *const *mut clocksource) };
let clock: *mut clocksource = __v0;
if static_branch_likely!(unsafe { core::ptr::addr_of_mut!(clocksource_read_inlined) }) != 0 {
return arch_inlined_clocksource_read(clock);
}
unsafe { unsafe { (*clock).read }.unwrap()(clock) }
}

// tk_clock_read - atomic clocksource read() helper
//
// This helper is necessary to use in the read paths because, while the
// seqcount ensures we don't return a bad value while structures are updated,
// it doesn't protect from potential crashes. There is the possibility that
// the tkr's clocksource may change between the read reference, and the
// clock reference passed to the read function.  This can cause crashes if
// the wrong clocksource is passed to the wrong read function.
// This isn't necessary to use when holding the tk_core.lock or doing
// a read of the fast-timekeeper tkrs (which is protected by its own locking
// and update logic).
#[inline(always)]
fn tk_clock_read__const6(tkr: *const tk_read_base) -> u64 {
let mut __v0: *mut clocksource;
while false {
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*tkr).clock) as *const *mut clocksource) };
let clock: *mut clocksource = __v0;
if static_branch_likely!(unsafe { core::ptr::addr_of_mut!(clocksource_read_inlined) }) != 0 {
return arch_inlined_clocksource_read(clock);
}
unsafe { unsafe { (*clock).read }.unwrap()(clock) }
}

fn clocksource_disable_inline_read() {
static_branch_disable!(unsafe { core::ptr::addr_of_mut!(clocksource_read_inlined) });
}

fn clocksource_enable_inline_read() {
static_branch_enable!(unsafe { core::ptr::addr_of_mut!(clocksource_read_inlined) });
}

/// tk_setup_internals - Set up internals to use clocksource clock.
///
/// @tk:		The target timekeeper to setup.
/// @clock:		Pointer to clocksource.
///
/// Calculates a fixed cycle/nsec interval for a given clocksource/adjustment
/// pair and interval request.
///
/// Unless you're the timekeeping code, you should not be using this!
fn tk_setup_internals(tk: *mut timekeeper, clock: *mut clocksource) {
let mut interval: u64 = 0;
unsafe {
(*tk).cs_was_changed_seq = ((*tk).cs_was_changed_seq as i32).wrapping_add(1) as u8;
}
let old_clock: *mut clocksource = unsafe { (*tk).tkr_mono.clock };
unsafe {
(*tk).tkr_mono.clock = clock;
}
unsafe {
(*tk).tkr_mono.mask = (*clock).mask;
}
unsafe {
(*tk).tkr_mono.cycle_last = tk_clock_read(core::ptr::addr_of_mut!((*tk).tkr_mono) as *const tk_read_base);
}
unsafe {
(*tk).tkr_raw.clock = clock;
}
unsafe {
(*tk).tkr_raw.mask = (*clock).mask;
}
unsafe {
(*tk).tkr_raw.cycle_last = (*tk).tkr_mono.cycle_last;
}
// Do the ns -> cycle conversion first, using original mult
interval = 1000000u64 << unsafe { (*clock).shift };
interval = interval.wrapping_add(((unsafe { (*clock).mult }) / 2) as u64);
do_div__2!(interval, unsafe { (*clock).mult });
if interval == 0 {
interval = 1;
}
unsafe {
(*tk).cycle_interval = interval;
}
// Go back from cycles -> shifted ns
unsafe {
(*tk).xtime_interval = interval.wrapping_mul((*clock).mult as u64);
}
unsafe {
(*tk).raw_interval = interval.wrapping_mul((*clock).mult as u64);
}
// if changing clocks, convert xtime_nsec shift units
if !old_clock.is_null() {
let shift_change: i32 = unsafe { (*clock).shift }.wrapping_sub(unsafe { (*old_clock).shift }) as i32;
if shift_change < 0 {
unsafe {
(*tk).tkr_mono.xtime_nsec = (*tk).tkr_mono.xtime_nsec >> shift_change.wrapping_neg();
}
unsafe {
(*tk).tkr_raw.xtime_nsec = (*tk).tkr_raw.xtime_nsec >> shift_change.wrapping_neg();
}
} else {
unsafe {
(*tk).tkr_mono.xtime_nsec = (*tk).tkr_mono.xtime_nsec << shift_change;
}
unsafe {
(*tk).tkr_raw.xtime_nsec = (*tk).tkr_raw.xtime_nsec << shift_change;
}
}
}
unsafe {
(*tk).tkr_mono.shift = (*clock).shift;
}
unsafe {
(*tk).tkr_raw.shift = (*clock).shift;
}
unsafe {
(*tk).ntp_error = 0;
}
unsafe {
(*tk).ntp_error_shift = 32u32.wrapping_sub((*clock).shift);
}
// ntp_tick is the tick length that NTP disciplines (its ±500 PPM
// scales only this part), in NTP-shifted ns: the real interval of
// a whole number of counter cycles. Because cycle_interval is
// rounded to an integer number of cycles, this ntp_tick differs
// from the true intended 1/HZ tick length by up to half a cycle
// period.
unsafe {
(*tk).ntp_tick = (*tk).xtime_interval << (*tk).ntp_error_shift;
}
// cs_tick_adj is the constant difference between the disciplined
// ntp_tick above and the true 1/HZ tick, expressed per-second to
// match the ntp_update_frequency() addends and handed to NTP via
// ntp_clear() to be explicitly included in its tick_length.
//
// Worked example: HZ=1000, ACPI PM timer at 3.579545 MHz, which
// has 3579.545 cycles in 1ms, rounded to cycle_interval = 3580.
//
// So ntp_tick is actually 1.000127ms, as that is the amount of
// time that 3580 cycles will take at the nominal frequency. This
// is the part that NTP disciplines, causing each 3580 counts to
// advance the clock by up to NTP's ±500PPM of that amount.
//
// The "extra" 127ns/tick is what's stored in cs_tick_adj and
// applied as a constant correction by ntp_update_frequency() so
// that NTP *believes* it's disciplining a 1ms tick.
unsafe {
(*tk).cs_tick_adj = ((*tk).ntp_tick as i64).wrapping_sub(4294967296000000i64);
}
let __v19: i64 = unsafe { (*tk).cs_tick_adj };
unsafe {
(*tk).cs_tick_adj = __v19.wrapping_mul(1000);
}
// The timekeeper keeps its own mult values for the currently
// active clocksource. These value will be adjusted via NTP
// to counteract clock drifting.
unsafe {
(*tk).tkr_mono.mult = (*clock).mult;
}
unsafe {
(*tk).tkr_raw.mult = (*clock).mult;
}
unsafe {
(*tk).ntp_err_mult = 0;
}
unsafe {
(*tk).skip_second_overflow = 0;
}
unsafe {
(*tk).skew_delta = 0;
}
unsafe {
(*tk).cs_id = (*clock).id;
}
// Coupled clockevent data
if IS_ENABLED__2!() != 0 && (unsafe { (*clock).flags }) & 1024 != 0 {
// Aim for an one hour maximum delta and use KHz to handle
// clocksources with a frequency above 4GHz correctly as
// the frequency argument of clocks_calc_mult_shift() is u32.
unsafe { clocks_calc_mult_shift(core::ptr::addr_of_mut!((*tk).cs_ns_to_cyc_mult), core::ptr::addr_of_mut!((*tk).cs_ns_to_cyc_shift), 1000000u32, (*clock).freq_khz, 3600000u32) };
// Initialize the conversion limit as the previous clocksource
// might have the same shift/mult pair so the quick check in
// tk_update_ns_to_cyc() fails to update it after a clocksource
// change leaving it effectivly zero.
unsafe {
(*tk).cs_ns_to_cyc_maxns = div_u64((*clock).mask, (*tk).cs_ns_to_cyc_mult);
}
}
}

/// Timekeeper helper functions.
#[inline(never)]
fn delta_to_ns_safe(tkr: *const tk_read_base, delta: u64) -> u64 {
mul_u64_u32_add_u64_shr(delta, unsafe { (*tkr).mult }, unsafe { (*tkr).xtime_nsec }, unsafe { (*tkr).shift })
}

#[inline(always)]
fn timekeeping_cycles_to_ns(tkr: *const tk_read_base, cycles: u64) -> u64 {
// Calculate the delta since the last update_wall_time()
let mask: u64 = unsafe { (*tkr).mask };
let delta: u64 = cycles.wrapping_sub(unsafe { (*tkr).cycle_last }) & mask;
// This detects both negative motion and the case where the delta
// overflows the multiplication with tkr->mult.
if unlikely__2!(delta > unsafe { (*(*tkr).clock).max_cycles }) != 0 {
// Handle clocksource inconsistency between CPUs to prevent
// time from going backwards by checking for the MSB of the
// mask being set in the delta.
if delta & !(mask >> 1i32) != 0 {
return (unsafe { (*tkr).xtime_nsec }) >> unsafe { (*tkr).shift };
}
return delta_to_ns_safe(tkr, delta);
}
delta.wrapping_mul((unsafe { (*tkr).mult }) as u64).wrapping_add(unsafe { (*tkr).xtime_nsec }) >> unsafe { (*tkr).shift }
}

#[inline(always)]
fn timekeeping_get_ns(tkr: *const tk_read_base) -> u64 {
timekeeping_cycles_to_ns(tkr, tk_clock_read(tkr))
}

#[inline(always)]
fn timekeeping_get_ns__const3(tkr: *const tk_read_base) -> u64 {
timekeeping_cycles_to_ns(tkr, tk_clock_read__const2(tkr))
}

#[inline(always)]
fn timekeeping_get_ns__const5(tkr: *const tk_read_base) -> u64 {
timekeeping_cycles_to_ns(tkr, tk_clock_read__const4(tkr))
}

/// update_fast_timekeeper - Update the fast and NMI safe monotonic timekeeper.
/// @tkr: Timekeeping readout base from which we take the update
/// @tkf: Pointer to NMI safe timekeeper
///
/// We want to use this from any context including NMI and tracing /
/// instrumenting the timekeeping code itself.
///
/// Employ the latch technique; see @write_seqcount_latch.
///
/// So if a NMI hits the update of base[0] then it will use base[1]
/// which is still consistent. In the worst case this can result is a
/// slightly wrong timestamp (a few nanoseconds). See
/// @ktime_get_mono_fast_ns.
fn update_fast_timekeeper(tkr: *const tk_read_base, tkf: *mut tk_fast) {
let base: *mut tk_read_base = unsafe { (*tkf).base.as_mut_ptr() as *mut tk_read_base };
// Force readers off to base[1]
write_seqcount_latch_begin(unsafe { core::ptr::addr_of_mut!((*tkf).seq) });
// Update base[0]
unsafe { memcpy(base as *mut (), tkr as *const (), core::mem::size_of::<tk_read_base>() as u64) };
// Force readers back to base[0]
write_seqcount_latch(unsafe { core::ptr::addr_of_mut!((*tkf).seq) });
// Update base[1]
unsafe { memcpy(base.wrapping_offset(1 as isize) as *mut (), base as *const (), core::mem::size_of::<tk_read_base>() as u64) };
write_seqcount_latch_end(unsafe { core::ptr::addr_of_mut!((*tkf).seq) });
}

#[inline(always)]
fn __ktime_get_fast_ns(tkf: *mut tk_fast) -> u64 {
let mut seq: u32 = 0;
let mut now: u64 = 0;
loop {
seq = read_seqcount_latch((unsafe { core::ptr::addr_of_mut!((*tkf).seq) }) as *const __SlateRecord_seqlock_h_612);
let tkr: *mut tk_read_base = unsafe { (*tkf).base.as_mut_ptr() as *mut tk_read_base }.wrapping_offset((seq & 1) as isize);
now = ktime_to_ns(unsafe { (*tkr).base }) as u64;
let __v0: u64 = timekeeping_get_ns(tkr as *const tk_read_base);
now = now.wrapping_add(__v0);
if read_seqcount_latch_retry((unsafe { core::ptr::addr_of_mut!((*tkf).seq) }) as *const __SlateRecord_seqlock_h_612, seq) == 0 {
break;
}
}
now
}

/// ktime_get_mono_fast_ns - Fast NMI safe access to clock monotonic
///
/// This timestamp is not guaranteed to be monotonic across an update.
/// The timestamp is calculated by:
///
/// 	now = base_mono + clock_delta * slope
///
/// So if the update lowers the slope, readers who are forced to the
/// not yet updated second array are still using the old steeper slope.
///
/// tmono
/// ^
/// |    o  n
/// |   o n
/// |  u
/// | o
/// |o
/// |12345678---> reader order
///
/// o = old slope
/// u = update
/// n = new slope
///
/// So reader 6 will observe time going backwards versus reader 5.
///
/// While other CPUs are likely to be able to observe that, the only way
/// for a CPU local observation is when an NMI hits in the middle of
/// the update. Timestamps taken from that NMI context might be ahead
/// of the following timestamps. Callers need to be aware of that and
/// deal with it.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_mono_fast_ns")]
extern "C" fn ktime_get_mono_fast_ns() -> u64 {
__ktime_get_fast_ns(unsafe { core::ptr::addr_of_mut!(tk_fast_mono.0) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_mono_fast_ns_518: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u64>, *mut ()>(Some(ktime_get_mono_fast_ns)) };

/// ktime_get_raw_fast_ns - Fast NMI safe access to clock monotonic raw
///
/// Contrary to ktime_get_mono_fast_ns() this is always correct because the
/// conversion factor is not affected by NTP/PTP correction.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_raw_fast_ns")]
extern "C" fn ktime_get_raw_fast_ns() -> u64 {
__ktime_get_fast_ns(unsafe { core::ptr::addr_of_mut!(tk_fast_raw.0) })
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_raw_fast_ns_519: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u64>, *mut ()>(Some(ktime_get_raw_fast_ns)) };

/// ktime_get_boot_fast_ns - NMI safe and fast access to boot clock.
///
/// To keep it NMI safe since we're accessing from tracing, we're not using a
/// separate timekeeper with updates to monotonic clock and boot offset
/// protected with seqcounts. This has the following minor side effects:
///
/// (1) Its possible that a timestamp be taken after the boot offset is updated
/// but before the timekeeper is updated. If this happens, the new boot offset
/// is added to the old timekeeping making the clock appear to update slightly
/// earlier:
///    CPU 0                                        CPU 1
///    timekeeping_inject_sleeptime64()
///    __timekeeping_inject_sleeptime(tk, delta);
///                                                 timestamp();
///    timekeeping_update_staged(tkd, TK_CLEAR_NTP...);
///
/// (2) On 32-bit systems, the 64-bit boot offset (tk->offs_boot) may be
/// partially updated.  Since the tk->offs_boot update is a rare event, this
/// should be a rare occurrence which postprocessing should be able to handle.
///
/// The caveats vs. timestamp ordering as documented for ktime_get_mono_fast_ns()
/// apply as well.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_boot_fast_ns")]
extern "C" fn ktime_get_boot_fast_ns() -> u64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
ktime_get_mono_fast_ns().wrapping_add(ktime_to_ns(data_race__4!(unsafe { (*tk).offs_boot })) as u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_boot_fast_ns_520: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u64>, *mut ()>(Some(ktime_get_boot_fast_ns)) };

/// ktime_get_tai_fast_ns - NMI safe and fast access to tai clock.
///
/// The same limitations as described for ktime_get_boot_fast_ns() apply. The
/// mono time and the TAI offset are not read atomically which may yield wrong
/// readouts. However, an update of the TAI offset is an rare event e.g., caused
/// by settime or adjtimex with an offset. The user of this function has to deal
/// with the possibility of wrong timestamps in post processing.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_tai_fast_ns")]
extern "C" fn ktime_get_tai_fast_ns() -> u64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
ktime_get_mono_fast_ns().wrapping_add(ktime_to_ns(data_race__4!(unsafe { (*tk).__slate_anon_6.offs_tai })) as u64)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_tai_fast_ns_521: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u64>, *mut ()>(Some(ktime_get_tai_fast_ns)) };

/// ktime_get_real_fast_ns: - NMI safe and fast access to clock realtime.
///
/// See ktime_get_mono_fast_ns() for documentation of the time stamp ordering.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_real_fast_ns")]
extern "C" fn ktime_get_real_fast_ns() -> u64 {
let tkf: *mut tk_fast = unsafe { core::ptr::addr_of_mut!(tk_fast_mono.0) };
let mut baser: u64 = 0;
let mut delta: u64 = 0;
let mut seq: u32 = 0;
loop {
seq = raw_read_seqcount_latch((unsafe { core::ptr::addr_of_mut!((*tkf).seq) }) as *const __SlateRecord_seqlock_h_612);
let tkr: *mut tk_read_base = unsafe { (*tkf).base.as_mut_ptr() as *mut tk_read_base }.wrapping_offset((seq & 1) as isize);
baser = ktime_to_ns((unsafe { (*tkr).base_real }) as i64) as u64;
delta = timekeeping_get_ns(tkr as *const tk_read_base);
if raw_read_seqcount_latch_retry((unsafe { core::ptr::addr_of_mut!((*tkf).seq) }) as *const __SlateRecord_seqlock_h_612, seq) == 0 {
break;
}
}
baser.wrapping_add(delta)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_real_fast_ns_522: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u64>, *mut ()>(Some(ktime_get_real_fast_ns)) };

/// halt_fast_timekeeper - Prevent fast timekeeper from accessing clocksource.
/// @tk: Timekeeper to snapshot.
///
/// It generally is unsafe to access the clocksource after timekeeping has been
/// suspended, so take a snapshot of the readout base of @tk and use it as the
/// fast timekeeper's readout base while suspended.  It will return the same
/// number of cycles every time until timekeeping is resumed at which time the
/// proper readout base for the fast timekeeper will be restored automatically.
fn halt_fast_timekeeper(tk: *const timekeeper) {
let mut tkr: *const tk_read_base = unsafe { core::ptr::addr_of!((*tk).tkr_mono) };
unsafe { memcpy(core::ptr::addr_of_mut!(tkr_dummy) as *mut (), tkr as *const (), core::mem::size_of::<tk_read_base>() as u64) };
unsafe {
cycles_at_suspend = tk_clock_read(tkr);
}
unsafe {
tkr_dummy.clock = core::ptr::addr_of_mut!(dummy_clock);
}
unsafe {
tkr_dummy.base_real = unsafe { (*tkr).base }.wrapping_add((*tk).offs_real) as u64;
}
update_fast_timekeeper((unsafe { core::ptr::addr_of_mut!(tkr_dummy) }) as *const tk_read_base, unsafe { core::ptr::addr_of_mut!(tk_fast_mono.0) });
tkr = unsafe { core::ptr::addr_of!((*tk).tkr_raw) };
unsafe { memcpy(core::ptr::addr_of_mut!(tkr_dummy) as *mut (), tkr as *const (), core::mem::size_of::<tk_read_base>() as u64) };
unsafe {
tkr_dummy.clock = core::ptr::addr_of_mut!(dummy_clock);
}
update_fast_timekeeper((unsafe { core::ptr::addr_of_mut!(tkr_dummy) }) as *const tk_read_base, unsafe { core::ptr::addr_of_mut!(tk_fast_raw.0) });
}

static mut tkr_dummy: tk_read_base = unsafe { core::mem::zeroed() };

static mut pvclock_gtod_chain: raw_notifier_head = raw_notifier_head { head: core::ptr::null_mut::<notifier_block>() };

fn update_pvclock_gtod(tk: *mut timekeeper, was_set: bool) {
unsafe { raw_notifier_call_chain(core::ptr::addr_of_mut!(pvclock_gtod_chain), was_set as u64, tk as *mut ()) };
}

/// pvclock_gtod_register_notifier - register a pvclock timedata update listener
/// @nb: Pointer to the notifier block to register
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.pvclock_gtod_register_notifier")]
extern "C" fn pvclock_gtod_register_notifier(nb: *mut notifier_block) -> i32 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut ret: i32 = 0;
let mut __UNIQUE_ID_guard_523: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_523) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_524: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_524) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
ret = unsafe { raw_notifier_chain_register(core::ptr::addr_of_mut!(pvclock_gtod_chain), nb) };
update_pvclock_gtod(tk, true);
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pvclock_gtod_register_notifier_525: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut notifier_block) -> i32>, *mut ()>(Some(pvclock_gtod_register_notifier)) };

/// pvclock_gtod_unregister_notifier - unregister a pvclock
/// timedata update listener
/// @nb: Pointer to the notifier block to unregister
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.pvclock_gtod_unregister_notifier")]
extern "C" fn pvclock_gtod_unregister_notifier(nb: *mut notifier_block) -> i32 {
let mut __UNIQUE_ID_guard_526: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_526) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_527: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_527) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe { raw_notifier_chain_unregister(core::ptr::addr_of_mut!(pvclock_gtod_chain), nb) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_pvclock_gtod_unregister_notifier_528: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut notifier_block) -> i32>, *mut ()>(Some(pvclock_gtod_unregister_notifier)) };

/// tk_update_leap_state - helper to update the next_leap_ktime
fn tk_update_leap_state(tk: *mut timekeeper) {
unsafe {
(*tk).next_leap_ktime = ntp_get_next_leap((*tk).id as u32);
}
if (unsafe { (*tk).next_leap_ktime }) != 9223372036854775807i64 {
unsafe {
(*tk).next_leap_ktime = ktime_sub!(unsafe { (*tk).next_leap_ktime }, unsafe { (*tk).offs_real });
}
}
// Convert to monotonic time
}

/// Leap state update for both shadow and the real timekeeper
/// Separate to spare a full memcpy() of the timekeeper.
fn tk_update_leap_state_all(tkd: *mut tk_data) {
loop {
seqprop_assert!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) });
if seqprop_preemptible!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }) {
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
do_write_seqcount_begin(seqprop_ptr!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }));
if true {
break;
}
}
tk_update_leap_state(unsafe { core::ptr::addr_of_mut!((*tkd).shadow_timekeeper) });
unsafe {
(*tkd).timekeeper.next_leap_ktime = (*tkd).shadow_timekeeper.next_leap_ktime;
}
loop {
do_write_seqcount_end(seqprop_ptr!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }));
if seqprop_preemptible!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }) {
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
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_529: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// Update the ktime_t based scalar nsec members of the timekeeper
fn tk_update_ktime_data(tk: *mut timekeeper) {
let mut seconds: u64 = 0;
let mut nsec: u32 = 0;
// The xtime based monotonic readout is:
// 	nsec = (xtime_sec + wtm_sec) * 1e9 + wtm_nsec + now();
// The ktime based monotonic readout is:
// 	nsec = base_mono + now();
// ==> base_mono = (xtime_sec + wtm_sec) * 1e9 + wtm_nsec
seconds = unsafe { (*tk).xtime_sec }.wrapping_add((unsafe { (*tk).wall_to_monotonic.tv_sec }) as u64);
nsec = (unsafe { (*tk).wall_to_monotonic.tv_nsec }) as u32;
unsafe {
(*tk).tkr_mono.base = ns_to_ktime(seconds.wrapping_mul(1000000000).wrapping_add(nsec as u64));
}
// The sum of the nanoseconds portions of xtime and
// wall_to_monotonic can be greater/equal one second. Take
// this into account before updating tk->ktime_sec.
nsec = nsec.wrapping_add(((unsafe { (*tk).tkr_mono.xtime_nsec }) >> unsafe { (*tk).tkr_mono.shift }) as u32);
if (nsec as i64) >= 1000000000 {
seconds = seconds.wrapping_add(1);
}
unsafe {
(*tk).ktime_sec = seconds;
}
// Update the monotonic raw base
unsafe {
(*tk).tkr_raw.base = ns_to_ktime(unsafe { (*tk).raw_sec }.wrapping_mul(1000000000));
}
}

fn tk_update_ns_to_cyc(tks: *mut timekeeper, tkc: *mut timekeeper) {
let tkrs: *mut tk_read_base = unsafe { core::ptr::addr_of_mut!((*tks).tkr_mono) };
let tkrc: *mut tk_read_base = unsafe { core::ptr::addr_of_mut!((*tkc).tkr_mono) };
let mut shift: u32 = 0;
if IS_ENABLED__2!() == 0 || (unsafe { (*(*tkrs).clock).flags }) & 1024 == 0 {
return;
}
if (unsafe { (*tkrs).mult }) == unsafe { (*tkrc).mult } && (unsafe { (*tkrs).shift }) == unsafe { (*tkrc).shift } {
return;
}
// The conversion math is simple:
//
//      CS::MULT       (1 << NS_TO_CYC_SHIFT)
//   --------------- = ----------------------
//   (1 << CS:SHIFT)       NS_TO_CYC_MULT
//
// Ergo:
//
//   NS_TO_CYC_MULT = (1 << (CS::SHIFT + NS_TO_CYC_SHIFT)) / CS::MULT
//
// NS_TO_CYC_SHIFT has been set up in tk_setup_internals()
shift = unsafe { (*tkrs).shift }.wrapping_add(unsafe { (*tks).cs_ns_to_cyc_shift });
unsafe {
(*tks).cs_ns_to_cyc_mult = div_u64(1u64 << shift, (*tkrs).mult) as u32;
}
unsafe {
(*tks).cs_ns_to_cyc_maxns = div_u64((*(*tkrs).clock).mask, (*tks).cs_ns_to_cyc_mult);
}
}

/// Restore the shadow timekeeper from the real timekeeper.
fn timekeeping_restore_shadow(tkd: *mut tk_data) {
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*tkd).lock) });
unsafe { memcpy(core::ptr::addr_of_mut!((*tkd).shadow_timekeeper) as *mut (), core::ptr::addr_of_mut!((*tkd).timekeeper) as *const (), core::mem::size_of::<timekeeper>() as u64) };
}

fn timekeeping_update_from_shadow(tkd: *mut tk_data, action: u32) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!((*tkd).shadow_timekeeper) };
lockdep_assert_held!(unsafe { core::ptr::addr_of_mut!((*tkd).lock) });
// Block out readers before running the updates below because that
// updates VDSO and other time related infrastructure. Not blocking
// the readers might let a reader see time going backwards when
// reading from the VDSO after the VDSO update and then reading in
// the kernel from the timekeeper before that got updated.
loop {
seqprop_assert!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) });
if seqprop_preemptible!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }) {
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
do_write_seqcount_begin(seqprop_ptr!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }));
if true {
break;
}
}
if action & 1u32 != 0 {
unsafe {
(*tk).ntp_error = 0;
}
unsafe { ntp_clear((*tk).id as u32, (*tk).cs_tick_adj) };
}
tk_update_leap_state(tk);
tk_update_ktime_data(tk);
unsafe {
(*tk).tkr_mono.base_real = unsafe { (*tk).tkr_mono.base }.wrapping_add((*tk).offs_real) as u64;
}
if ((unsafe { (*tk).id }) as u32) == 0 {
tk_update_ns_to_cyc(tk, unsafe { core::ptr::addr_of_mut!((*tkd).timekeeper) });
unsafe { update_vsyscall(tk) };
update_pvclock_gtod(tk, action & 2u32 != 0);
update_fast_timekeeper((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base, unsafe { core::ptr::addr_of_mut!(tk_fast_mono.0) });
update_fast_timekeeper((unsafe { core::ptr::addr_of_mut!((*tk).tkr_raw) }) as *const tk_read_base, unsafe { core::ptr::addr_of_mut!(tk_fast_raw.0) });
} else {
if tk_is_aux(tk as *const timekeeper) {
unsafe { vdso_time_update_aux(tk) };
}
}
if action & 2u32 != 0 {
let __v4: u32 = unsafe { (*tk).clock_was_set_seq };
unsafe {
(*tk).clock_was_set_seq = __v4.wrapping_add(1);
}
}
// Update the real timekeeper.
//
// We could avoid this memcpy() by switching pointers, but that has
// the downside that the reader side does not longer benefit from
// the cacheline optimized data layout of the timekeeper and requires
// another indirection.
//
// Write xtime_sec first so that even if the memcpy() tears the store
// data integrity is provided for ktime_get_real_seconds().
loop {
loop {
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_530() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__2!((*tkd).timekeeper.xtime_sec, unsafe { (*tk).xtime_sec });
if true {
break;
}
}
unsafe { memcpy(core::ptr::addr_of_mut!((*tkd).timekeeper) as *mut (), tk as *const (), core::mem::size_of::<timekeeper>() as u64) };
loop {
do_write_seqcount_end(seqprop_ptr!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }));
if seqprop_preemptible!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }) {
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
if true {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_531: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

/// timekeeping_forward_now - update clock to the current time
/// @tk:		Pointer to the timekeeper to update
///
/// Forward the current clock to update its state since the last call to
/// update_wall_time(). This is useful before significant clock changes,
/// as it avoids having to deal with this time offset explicitly.
fn timekeeping_forward_now(tk: *mut timekeeper) {
let cycle_now: u64 = tk_clock_read((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base);
let mut delta: u64 = clocksource_delta(cycle_now, unsafe { (*tk).tkr_mono.cycle_last }, unsafe { (*tk).tkr_mono.mask }, unsafe { (*(*tk).tkr_mono.clock).max_raw_delta });
unsafe {
(*tk).tkr_mono.cycle_last = cycle_now;
}
unsafe {
(*tk).tkr_raw.cycle_last = cycle_now;
}
while delta > 0 {
let max: u64 = unsafe { (*(*tk).tkr_mono.clock).max_cycles };
let incr: u64 = if delta < max { delta } else { max };
let __v1: u64 = unsafe { (*tk).tkr_mono.xtime_nsec };
unsafe {
(*tk).tkr_mono.xtime_nsec = __v1.wrapping_add(incr.wrapping_mul((*tk).tkr_mono.mult as u64));
}
let __v4: u64 = unsafe { (*tk).tkr_raw.xtime_nsec };
unsafe {
(*tk).tkr_raw.xtime_nsec = __v4.wrapping_add(incr.wrapping_mul((*tk).tkr_raw.mult as u64));
}
tk_normalize_xtime(tk);
delta = delta.wrapping_sub(incr);
}
tk_update_coarse_nsecs(tk);
}

// ktime_expiry_to_cycles - Convert a expiry time to clocksource cycles
// @id:		Clocksource ID which is required for validity
// @expires_ns:	Absolute CLOCK_MONOTONIC expiry time (nsecs) to be converted
// @cycles:	Pointer to storage for corresponding absolute cycles value
//
// Convert a CLOCK_MONOTONIC based absolute expiry time to a cycles value
// based on the correlated clocksource of the clockevent device by using
// the base nanoseconds and cycles values of the last timekeeper update and
// converting the delta between @expires_ns and base nanoseconds to cycles.
//
// This only works for clockevent devices which are using a less than or
// equal comparator against the clocksource.
//
// Utilizing this avoids two clocksource reads for such devices, the
// ktime_get() in clockevents_program_event() to calculate the delta expiry
// value and the readout in the device::set_next_event() callback to
// convert the delta back to a absolute comparator value.
//
// Returns: True if @id matches the current clocksource ID, false otherwise
/// ktime_get_real_ts64 - Returns the time of day in a timespec64.
/// @ts:		pointer to the timespec to be set
///
/// Returns the time of day in a timespec64 (WARN if suspended).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_real_ts64")]
extern "C" fn ktime_get_real_ts64(ts: *mut timespec64) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut nsecs: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { timekeeping_suspended }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("538: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 538b - ., 3; .popsection\n# {0}", const 538 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_real_ts64_1, sym __slate_asm_string_ktime_get_real_ts64_2, const 977 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("539: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 539b - ., 4; .popsection\n# {0}", const 539 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
unsafe {
(*ts).tv_sec = (*tk).xtime_sec as i64;
}
nsecs = timekeeping_get_ns__const5((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base);
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
unsafe {
(*ts).tv_nsec = 0;
}
timespec64_add_ns(ts, nsecs);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_real_ts64_540: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut timespec64)>, *mut ()>(Some(ktime_get_real_ts64)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get")]
extern "C" fn ktime_get() -> i64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut base: i64 = 0;
let mut nsecs: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { timekeeping_suspended }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("541: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 541b - ., 3; .popsection\n# {0}", const 541 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_1, sym __slate_asm_string_ktime_get_2, const 999 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("542: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 542b - ., 4; .popsection\n# {0}", const 542 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
base = unsafe { (*tk).tkr_mono.base };
nsecs = timekeeping_get_ns__const5((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base);
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
ktime_add_ns!(base, nsecs) as i64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_543: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i64>, *mut ()>(Some(ktime_get)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_resolution_ns")]
extern "C" fn ktime_get_resolution_ns() -> u32 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut nsecs: u32 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { timekeeping_suspended }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("544: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 544b - ., 3; .popsection\n# {0}", const 544 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_resolution_ns_1, sym __slate_asm_string_ktime_get_resolution_ns_2, const 1018 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("545: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 545b - ., 4; .popsection\n# {0}", const 545 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
nsecs = (unsafe { (*tk).tkr_mono.mult }) >> unsafe { (*tk).tkr_mono.shift };
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
nsecs
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_resolution_ns_546: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u32>, *mut ()>(Some(ktime_get_resolution_ns)) };

static offsets: __SlateAlign16<[*const i64; 3]> = __SlateAlign16([(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0i32 as usize].timekeeper.offs_real) }) as *const i64, (unsafe { core::ptr::addr_of_mut!(timekeeper_data[0i32 as usize].timekeeper.offs_boot) }) as *const i64, (unsafe { core::ptr::addr_of_mut!(timekeeper_data[0i32 as usize].timekeeper.__slate_anon_6.offs_tai) }) as *const i64]);

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_with_offset")]
extern "C" fn ktime_get_with_offset(offs: u32) -> i64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let offset: *const i64 = unsafe { offsets.0[(offs as u32) as usize] };
let mut seq: u32 = 0;
let mut base: i64 = 0;
let mut nsecs: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { timekeeping_suspended }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("547: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 547b - ., 3; .popsection\n# {0}", const 547 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_with_offset_1, sym __slate_asm_string_ktime_get_with_offset_2, const 1043 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("548: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 548b - ., 4; .popsection\n# {0}", const 548 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
base = ktime_add!(unsafe { (*tk).tkr_mono.base }, unsafe { *offset });
nsecs = timekeeping_get_ns__const5((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base);
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
ktime_add_ns!(base, nsecs) as i64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_with_offset_549: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> i64>, *mut ()>(Some(ktime_get_with_offset)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_coarse_with_offset")]
extern "C" fn ktime_get_coarse_with_offset(offs: u32) -> i64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let offset: *const i64 = unsafe { offsets.0[(offs as u32) as usize] };
let mut seq: u32 = 0;
let mut base: i64 = 0;
let mut nsecs: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { timekeeping_suspended }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("550: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 550b - ., 3; .popsection\n# {0}", const 550 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_coarse_with_offset_1, sym __slate_asm_string_ktime_get_coarse_with_offset_2, const 1065 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("551: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 551b - ., 4; .popsection\n# {0}", const 551 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
base = ktime_add!(unsafe { (*tk).tkr_mono.base }, unsafe { *offset });
nsecs = (unsafe { (*tk).coarse_nsec }) as u64;
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
ktime_add_ns!(base, nsecs) as i64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_coarse_with_offset_552: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> i64>, *mut ()>(Some(ktime_get_coarse_with_offset)) };

// ktime_mono_to_any() - convert monotonic time to any other time
// @tmono:	time to convert.
// @offs:	which offset to use
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_mono_to_any_554: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i64, u32) -> i64>, *mut ()>(Some(ktime_mono_to_any)) };

/// ktime_get_raw - Returns the raw monotonic time in ktime_t format
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_raw")]
extern "C" fn ktime_get_raw() -> i64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut base: i64 = 0;
let mut nsecs: u64 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
base = unsafe { (*tk).tkr_raw.base };
nsecs = timekeeping_get_ns__const3((unsafe { core::ptr::addr_of_mut!((*tk).tkr_raw) }) as *const tk_read_base);
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
ktime_add_ns!(base, nsecs) as i64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_raw_555: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i64>, *mut ()>(Some(ktime_get_raw)) };

/// ktime_get_ts64 - get the monotonic clock in timespec64 format
/// @ts:		pointer to timespec variable
///
/// The function calculates the monotonic clock from the realtime
/// clock and the wall_to_monotonic offset and stores the result
/// in normalized timespec64 format in the variable pointed to by @ts.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_ts64")]
extern "C" fn ktime_get_ts64(ts: *mut timespec64) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut tomono: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tomono) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut seq: u32 = 0;
let mut nsec: u64 = 0;
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { timekeeping_suspended }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("556: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 556b - ., 3; .popsection\n# {0}", const 556 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_ts64_1, sym __slate_asm_string_ktime_get_ts64_2, const 1142 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("557: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 557b - ., 4; .popsection\n# {0}", const 557 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
unsafe {
(*ts).tv_sec = (*tk).xtime_sec as i64;
}
nsec = timekeeping_get_ns__const5((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base);
tomono = unsafe { (*tk).wall_to_monotonic };
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
let __v12: i64 = unsafe { (*ts).tv_sec };
unsafe {
(*ts).tv_sec = __v12.wrapping_add(tomono.tv_sec);
}
unsafe {
(*ts).tv_nsec = 0;
}
timespec64_add_ns(ts, nsec.wrapping_add(tomono.tv_nsec as u64));
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_ts64_558: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut timespec64)>, *mut ()>(Some(ktime_get_ts64)) };

/// ktime_get_seconds - Get the seconds portion of CLOCK_MONOTONIC
///
/// Returns the seconds portion of CLOCK_MONOTONIC with a single non
/// serialized read. tk->ktime_sec is of type 'unsigned long' so this
/// works on both 32 and 64 bit systems. On 32 bit systems the readout
/// covers ~136 years of uptime which should be enough to prevent
/// premature wrap arounds.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_seconds")]
extern "C" fn ktime_get_seconds() -> i64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { timekeeping_suspended }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("559: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 559b - ., 3; .popsection\n# {0}", const 559 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_seconds_1, sym __slate_asm_string_ktime_get_seconds_2, const 1171 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("560: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 560b - ., 4; .popsection\n# {0}", const 560 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
(unsafe { (*tk).ktime_sec }) as i64
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_seconds_561: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i64>, *mut ()>(Some(ktime_get_seconds)) };

// ktime_get_real_seconds - Get the seconds portion of CLOCK_REALTIME
//
// Returns the wall clock seconds since 1970.
//
// For 64bit systems the fast access to tk->xtime_sec is preserved. On
// 32bit systems the access must be protected with the sequence
// counter to provide "atomic" access to the 64bit tk->xtime_sec
// value.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_real_seconds_564: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i64>, *mut ()>(Some(ktime_get_real_seconds)) };

// __ktime_get_real_seconds - Unprotected access to CLOCK_REALTIME seconds
//
// The same as ktime_get_real_seconds() but without the sequence counter
// protection. This function is used in restricted contexts like the x86 MCE
// handler and in KGDB. It's unprotected on 32-bit vs. concurrent half
// completed modification and only to be used for such critical contexts.
//
// Returns: Racy snapshot of the CLOCK_REALTIME seconds value
fn tk_clock_read_snapshot(tkr: *const tk_read_base, chs: *mut clocksource_hw_snapshot) -> u64 {
let mut __v0: *mut clocksource;
loop {
if !(__native_word__57!() || ((const { core::mem::size_of::<*mut clocksource>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_566() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*tkr).clock) as *const *mut clocksource) };
let clock: *mut clocksource = __v0;
if unlikely__6!(unsafe { (*clock).read_snapshot }) != 0 {
return unsafe { unsafe { (*clock).read_snapshot }.unwrap()(clock, chs) };
}
unsafe { unsafe { (*clock).read }.unwrap()(clock) }
}

/// ktime_get_snapshot_id -  Simultaneously snapshot a given clock ID with
/// 			    the corresponding monotonic raw and the underlying
/// 			    clocksource counter value.
/// @clock_id:		The clock ID to snapshot
/// @systime_snapshot:	Pointer to struct receiving the system time snapshot
///
/// For the system time keeping clocks (REALTIME, MONOTONIC and BOOTTIME) the
/// monotonic raw clock is CLOCK_MONOTONIC_RAW. For AUX clocks this is the
/// monotonic raw clock related to the AUX clock. These AUX clock related
/// monotonic raw clocks have a strict linear offset to the system time
/// CLOCK_MONOTONIC_RAW:
///
/// 	MONOTONIC_RAW(AUX$N) = CLOCK_MONOTONIC_RAW(system) + offset(AUX$N)
///
/// The offset is established when a AUX clock is initialized, but it is
/// currently not accessible.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_snapshot_id")]
extern "C" fn ktime_get_snapshot_id(clock_id: i32, systime_snapshot: *mut system_time_snapshot) {
let mut base_raw: i64 = 0;
let mut base_sys: i64 = 0;
let mut offs_sys: i64 = 0;
let mut offs: *mut i64 = unsafe { core::mem::zeroed() };
let mut offs_zero: i64 = 0;
let mut nsec_raw: u64 = 0;
let mut nsec_sys: u64 = 0;
let mut now: u64 = 0;
let mut tk: *mut timekeeper = unsafe { core::mem::zeroed() };
let mut tkd: *mut tk_data = unsafe { core::mem::zeroed() };
let mut seq: u32 = 0;
// Invalidate the snapshot for all failure cases
unsafe {
(*systime_snapshot).valid = 0;
}
let mut __v0: i64;
{
let __ret_warn_on: i32 = ((unsafe { timekeeping_suspended }) != 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("567: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 567b - ., 3; .popsection\n# {0}", const 567 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_snapshot_id_1, sym __slate_asm_string_ktime_get_snapshot_id_2, const 1263 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("568: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 568b - ., 4; .popsection\n# {0}", const 568 as i32, options(nostack, att_syntax));
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
'__slate_break_5: {
match clock_id {
0 => {
tkd = unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize);
offs = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper.offs_real) };
break '__slate_break_5;
// Map RAW to MONOTONIC so the loop below is trivial
}
1 | 4 => {
tkd = unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize);
offs = core::ptr::addr_of_mut!(offs_zero);
}
7 => {
tkd = unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize);
offs = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper.offs_boot) };
}
16..=23 => {
tkd = aux_get_tk_data(clock_id);
if tkd.is_null() {
return;
}
offs = unsafe { core::ptr::addr_of_mut!((*tkd).timekeeper.__slate_anon_6.offs_aux) };
}
_ => {
let mut __v6: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("569: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 569b - ., 3; .popsection\n# {0}", const 569 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_ktime_get_snapshot_id_7, sym __slate_asm_string_ktime_get_snapshot_id_8, const 1288 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("570: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 570b - ., 4; .popsection\n# {0}", const 570 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v6 = (__ret_warn_on != 0) as i64;
return;
}
}
}
tk = unsafe { core::ptr::addr_of_mut!((*tkd).timekeeper) };
'__slate_break_16: loop {
let mut chs: clocksource_hw_snapshot = {
let mut __t0: clocksource_hw_snapshot = unsafe { core::mem::zeroed() };
__t0
};
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) });
// Aux clocks can be invalid
if (unsafe { (*tk).clock_valid }) == 0 {
return;
}
now = tk_clock_read_snapshot((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base, core::ptr::addr_of_mut!(chs));
unsafe {
(*systime_snapshot).cs_id = (*(*tk).tkr_mono.clock).id;
}
unsafe {
(*systime_snapshot).hw_cycles = chs.hw_cycles;
}
unsafe {
(*systime_snapshot).hw_csid = chs.hw_csid;
}
unsafe {
(*systime_snapshot).cs_was_changed_seq = (*tk).cs_was_changed_seq;
}
unsafe {
(*systime_snapshot).clock_was_set_seq = (*tk).clock_was_set_seq;
}
base_sys = unsafe { (*tk).tkr_mono.base };
offs_sys = unsafe { *offs };
base_raw = unsafe { (*tk).tkr_raw.base };
nsec_sys = timekeeping_cycles_to_ns((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base, now);
nsec_raw = timekeeping_cycles_to_ns((unsafe { core::ptr::addr_of_mut!((*tk).tkr_raw) }) as *const tk_read_base, now);
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }, seq) == 0 {
break;
}
}
unsafe {
(*systime_snapshot).cycles = now;
}
unsafe {
(*systime_snapshot).systime = ktime_add_ns!(base_sys, (offs_sys as u64).wrapping_add(nsec_sys)) as i64;
}
unsafe {
(*systime_snapshot).monoraw = ktime_add_ns!(base_raw, nsec_raw) as i64;
}
// Special case for PTP. Just transfer the raw time into sys,
// so the call sites can consistently use snap::systime.
if clock_id == 4 {
unsafe {
(*systime_snapshot).systime = (*systime_snapshot).monoraw;
}
}
// Tell the consumer that this snapshot is valid
unsafe {
(*systime_snapshot).valid = 1;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_snapshot_id_571: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, *mut system_time_snapshot)>, *mut ()>(Some(ktime_get_snapshot_id)) };

/// Scale base by mult/div checking for overflow
fn scale64_check_overflow(mult: u64, div: u64, base: &mut u64) -> i32 {
let mut tmp: u64 = 0;
let mut rem: u64 = 0;
tmp = div64_u64_rem(unsafe { *(base as *mut u64) }, div, unsafe { &mut (*core::ptr::addr_of_mut!(rem)) });
if (core::mem::size_of::<u64>() as i32).wrapping_mul(8).wrapping_sub(fls64(mult)) < fls64(tmp) || (core::mem::size_of::<u64>() as i32).wrapping_mul(8).wrapping_sub(fls64(mult)) < fls64(rem) {
return 75i32.wrapping_neg();
}
tmp = tmp.wrapping_mul(mult);
rem = div64_u64(rem.wrapping_mul(mult), div);
unsafe {
*(base as *mut u64) = tmp.wrapping_add(rem);
}
0
}

/// adjust_historical_crosststamp - adjust crosstimestamp previous to current interval
/// @history:			Snapshot representing start of history
/// @partial_history_cycles:	Cycle offset into history (fractional part)
/// @total_history_cycles:	Total history length in cycles
/// @discontinuity:		True indicates clock was set on history period
/// @ts:				Cross timestamp that should be adjusted using
/// 	partial/total ratio
///
/// Helper function used by get_device_system_crosststamp() to correct the
/// crosstimestamp corresponding to the start of the current interval to the
/// system counter value (timestamp point) provided by the driver. The
/// total_history_* quantities are the total history starting at the provided
/// reference point and ending at the start of the current interval. The cycle
/// count between the driver timestamp point and the start of the current
/// interval is partial_history_cycles.
fn adjust_historical_crosststamp(history: *mut system_time_snapshot, mut partial_history_cycles: u64, total_history_cycles: u64, discontinuity: bool, ts: *mut system_device_crosststamp) -> i32 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut corr_raw: u64 = 0;
let mut corr_sys: u64 = 0;
let mut interp_forward: bool = false;
let mut ret: i32 = 0;
if total_history_cycles == 0 || partial_history_cycles == 0 {
return 0;
}
// Interpolate shortest distance from beginning or end of history
interp_forward = partial_history_cycles > total_history_cycles / 2;
partial_history_cycles = if interp_forward { total_history_cycles.wrapping_sub(partial_history_cycles) } else { partial_history_cycles };
// Scale the monotonic raw time delta by:
// 	partial_history_cycles / total_history_cycles
corr_raw = ktime_to_ns(ktime_sub!(unsafe { (*ts).sys_monoraw }, unsafe { (*history).monoraw })) as u64;
ret = scale64_check_overflow(partial_history_cycles, total_history_cycles, unsafe { &mut (*core::ptr::addr_of_mut!(corr_raw)) });
if ret != 0 {
return ret;
}
// If there is a discontinuity in the history, scale monotonic raw
// correction by:
// 	mult(sys)/mult(raw) yielding the system time correction
//
// Otherwise, calculate the system time correction similar to monotonic
// raw calculation
if discontinuity {
corr_sys = mul_u64_u32_div(corr_raw, unsafe { (*tk).tkr_mono.mult }, unsafe { (*tk).tkr_raw.mult });
} else {
corr_sys = ktime_to_ns(ktime_sub!(unsafe { (*ts).sys_systime }, unsafe { (*history).systime })) as u64;
ret = scale64_check_overflow(partial_history_cycles, total_history_cycles, unsafe { &mut (*core::ptr::addr_of_mut!(corr_sys)) });
if ret != 0 {
return ret;
}
}
// Fixup monotonic raw and system time time values
if interp_forward {
unsafe {
(*ts).sys_monoraw = ktime_add_ns!(unsafe { (*history).monoraw }, corr_raw) as i64;
}
unsafe {
(*ts).sys_systime = ktime_add_ns!(unsafe { (*history).systime }, corr_sys) as i64;
}
} else {
unsafe {
(*ts).sys_monoraw = ktime_sub_ns!(unsafe { (*ts).sys_monoraw }, corr_raw) as i64;
}
unsafe {
(*ts).sys_systime = ktime_sub_ns!(unsafe { (*ts).sys_systime }, corr_sys) as i64;
}
}
0
}

/// timestamp_in_interval - true if ts is chronologically in [start, end]
///
/// True if ts occurs chronologically at or after start, and before or at end.
fn timestamp_in_interval(start: u64, end: u64, ts: u64) -> bool {
if ts >= start && ts <= end {
return true;
}
if start > end && (ts >= start || ts <= end) {
return true;
}
false
}

fn convert_clock(val: *mut u64, numerator: u32, denominator: u32) -> bool {
let mut rem: u64 = 0;
let mut res: u64 = 0;
if numerator == 0 || denominator == 0 {
return false;
}
res = div64_u64_rem(unsafe { *val }, denominator as u64, unsafe { &mut (*core::ptr::addr_of_mut!(rem)) }).wrapping_mul(numerator as u64);
unsafe {
*val = res.wrapping_add(div_u64(rem.wrapping_mul(numerator as u64), denominator));
}
true
}

fn convert_base_to_cs(scv: *mut system_counterval_t) -> bool {
let cs: *mut clocksource = unsafe { timekeeper_data[0].timekeeper.tkr_mono.clock };
let mut num: u32 = 0;
let mut den: u32 = 0;
// The timestamp was taken from the time keeper clock source
if ((unsafe { (*cs).id }) as u32) == ((unsafe { (*scv).cs_id }) as u32) {
return true;
}
// Check whether cs_id matches the base clock. Prevent the compiler from
// re-evaluating @base as the clocksource might change concurrently.
let mut __v0: *mut clocksource_base;
loop {
if !(__native_word__44!() || ((const { core::mem::size_of::<*mut clocksource_base>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_572() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*cs).base) as *const *mut clocksource_base) };
let base: *mut clocksource_base = __v0;
if base.is_null() || ((unsafe { (*base).id }) as u32) != ((unsafe { (*scv).cs_id }) as u32) {
return false;
}
num = if unsafe { (*scv).use_nsecs } { unsafe { (*cs).freq_khz } } else { unsafe { (*base).numerator } };
den = (if unsafe { (*scv).use_nsecs } { 1000000i64 } else { (unsafe { (*base).denominator }) as i64 }) as u32;
if !convert_clock(unsafe { core::ptr::addr_of_mut!((*scv).cycles) }, num, den) {
return false;
}
let __v3: u64 = unsafe { (*scv).cycles };
unsafe {
(*scv).cycles = __v3.wrapping_add((*base).offset);
}
// Set the clocksource ID as scv::cycles is now clocksource based
unsafe {
(*scv).cs_id = (*cs).id;
}
true
}

fn convert_cs_to_base(cycles: *mut u64, base_id: u32) -> bool {
let cs: *mut clocksource = unsafe { timekeeper_data[0].timekeeper.tkr_mono.clock };
// Check whether base_id matches the base clock. Prevent the compiler from
// re-evaluating @base as the clocksource might change concurrently.
let mut __v0: *mut clocksource_base;
loop {
if !(__native_word__44!() || ((const { core::mem::size_of::<*mut clocksource_base>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_573() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*cs).base) as *const *mut clocksource_base) };
let base: *mut clocksource_base = __v0;
if base.is_null() || ((unsafe { (*base).id }) as u32) != (base_id as u32) {
return false;
}
let __v3: u64 = unsafe { *cycles };
unsafe {
*cycles = __v3.wrapping_sub((*base).offset);
}
if !convert_clock(cycles, unsafe { (*base).denominator }, unsafe { (*base).numerator }) {
return false;
}
true
}

fn convert_ns_to_cs(delta: *mut u64) -> bool {
let tkr: *mut tk_read_base = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper.tkr_mono) };
if BITS_TO_BYTES!((fls64(unsafe { *delta }) as u32).wrapping_add(unsafe { (*tkr).shift })) >= (core::mem::size_of::<u64>() as u64) {
return false;
}
unsafe {
*delta = div_u64((*delta << (*tkr).shift).wrapping_sub((*tkr).xtime_nsec), (*tkr).mult);
}
true
}

/// ktime_real_to_base_clock() - Convert CLOCK_REALTIME timestamp to a base clock timestamp
/// @treal:	CLOCK_REALTIME timestamp to convert
/// @base_id:	base clocksource id
/// @cycles:	pointer to store the converted base clock timestamp
///
/// Converts a supplied, future realtime clock value to the corresponding base clock value.
///
/// Return:  true if the conversion is successful, false otherwise.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_real_to_base_clock")]
extern "C" fn ktime_real_to_base_clock(treal: i64, base_id: u32, cycles: *mut u64) -> bool {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut delta: u64 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
if (treal as u64) < unsafe { (*tk).tkr_mono.base_real } {
return false;
}
delta = (treal as u64).wrapping_sub(unsafe { (*tk).tkr_mono.base_real });
if !convert_ns_to_cs(core::ptr::addr_of_mut!(delta)) {
return false;
}
unsafe {
*cycles = unsafe { (*tk).tkr_mono.cycle_last }.wrapping_add(delta);
}
if !convert_cs_to_base(cycles, base_id) {
return false;
}
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
true
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_real_to_base_clock_574: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i64, u32, *mut u64) -> bool>, *mut ()>(Some(ktime_real_to_base_clock)) };

/// get_device_system_crosststamp - Synchronously capture system/device timestamp
/// @get_time_fn:	Callback to get simultaneous device time and system counter
/// 			from the device driver
/// @ctx:		Context passed to get_time_fn()
/// @history_begin:	Historical reference point used to interpolate system time when
/// 			the counter value provided by the driver is before the current interval
/// @xtstamp:		Receives simultaneously captured system and device time
///
/// Reads a timestamp from a device and correlates it to system time
///
/// See documentation for ktime_get_snapshot_id() for information about the raw
/// monotonic time stamp which is used here.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.get_device_system_crosststamp")]
extern "C" fn get_device_system_crosststamp(get_time_fn: Option<unsafe extern "C" fn(*mut i64, *mut system_counterval_t, *mut ()) -> i32>, ctx: *mut (), history_begin: *mut system_time_snapshot, xtstamp: *mut system_device_crosststamp) -> i32 {
let mut syscnt_cycles: u64 = 0;
let mut cycles: u64 = 0;
let mut now: u64 = 0;
let mut interval_start: u64 = 0;
let mut base_sys: i64 = 0;
let mut base_raw: i64 = 0;
let mut offs: *mut i64 = unsafe { core::mem::zeroed() };
let mut clock_was_set_seq: u32 = 0;
let mut nsec_sys: u64 = 0;
let mut nsec_raw: u64 = 0;
let mut cs_was_changed_seq: u8 = 0;
let mut seq: u32 = 0;
let mut do_interp: bool = false;
let mut tk: *mut timekeeper = unsafe { core::mem::zeroed() };
let mut tkd: *mut tk_data = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
match unsafe { (*xtstamp).clock_id } {
0 => {
tkd = unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize);
offs = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper.offs_real) };
}
16..=23 => {
tkd = aux_get_tk_data(unsafe { (*xtstamp).clock_id });
if tkd.is_null() {
return 19i32.wrapping_neg();
}
offs = unsafe { core::ptr::addr_of_mut!((*tkd).timekeeper.__slate_anon_6.offs_aux) };
}
_ => {
let mut __v1: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("575: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 575b - ., 3; .popsection\n# {0}", const 575 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_get_device_system_crosststamp_2, sym __slate_asm_string_get_device_system_crosststamp_3, const 1592 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("576: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 576b - ., 4; .popsection\n# {0}", const 576 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
return 19i32.wrapping_neg();
}
}
tk = unsafe { core::ptr::addr_of_mut!((*tkd).timekeeper) };
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) });
// Try to synchronously capture device time and a system
// counter value calling back into the device driver
ret = unsafe { get_time_fn.unwrap()(core::ptr::addr_of_mut!((*xtstamp).device), core::ptr::addr_of_mut!((*xtstamp).sys_counter), ctx) };
if ret != 0 {
return ret;
}
// Verify that the clocksource ID associated with the captured
// system counter value is the same as for the currently
// installed timekeeper clocksource and convert to it.
let __v11: bool = ((unsafe { (*xtstamp).sys_counter.cs_id }) as u32) == 0 || !convert_base_to_cs(unsafe { core::ptr::addr_of_mut!((*xtstamp).sys_counter) });
if __v11 {
return 19i32.wrapping_neg();
}
let __v12: u64 = unsafe { (*xtstamp).sys_counter.cycles };
syscnt_cycles = __v12;
cycles = __v12;
// Check whether the system counter value provided by the
// device driver is on the current timekeeping interval.
now = tk_clock_read((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base);
interval_start = unsafe { (*tk).tkr_mono.cycle_last };
if !timestamp_in_interval(interval_start, now, cycles) {
clock_was_set_seq = unsafe { (*tk).clock_was_set_seq };
cs_was_changed_seq = unsafe { (*tk).cs_was_changed_seq };
cycles = interval_start;
do_interp = true;
} else {
do_interp = false;
}
base_sys = ktime_add!(unsafe { (*tk).tkr_mono.base }, unsafe { *offs });
base_raw = unsafe { (*tk).tkr_raw.base };
nsec_sys = timekeeping_cycles_to_ns((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base, cycles);
nsec_raw = timekeeping_cycles_to_ns((unsafe { core::ptr::addr_of_mut!((*tk).tkr_raw) }) as *const tk_read_base, cycles);
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) }, seq) == 0 {
break;
}
}
unsafe {
(*xtstamp).sys_systime = ktime_add_ns!(base_sys, nsec_sys) as i64;
}
unsafe {
(*xtstamp).sys_monoraw = ktime_add_ns!(base_raw, nsec_raw) as i64;
}
// Interpolate if necessary, adjusting back from the start of the
// current interval
if do_interp {
let mut partial_history_cycles: u64 = 0;
let mut total_history_cycles: u64 = 0;
let mut discontinuity: bool = false;
// Check that the counter value is not before the provided
// history reference and that the history doesn't cross a
// clocksource change
let __v14: bool = history_begin.is_null() || !timestamp_in_interval(unsafe { (*history_begin).cycles }, cycles, syscnt_cycles);
if __v14 || ((unsafe { (*history_begin).cs_was_changed_seq }) as i32) != (cs_was_changed_seq as i32) {
return 22i32.wrapping_neg();
}
partial_history_cycles = cycles.wrapping_sub(syscnt_cycles);
total_history_cycles = cycles.wrapping_sub(unsafe { (*history_begin).cycles });
discontinuity = (unsafe { (*history_begin).clock_was_set_seq }) != clock_was_set_seq;
ret = adjust_historical_crosststamp(history_begin, partial_history_cycles, total_history_cycles, discontinuity, xtstamp);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_get_device_system_crosststamp_577: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(Option<unsafe extern "C" fn(*mut i64, *mut system_counterval_t, *mut ()) -> i32>, *mut (), *mut system_time_snapshot, *mut system_device_crosststamp) -> i32>, *mut ()>(Some(get_device_system_crosststamp)) };

// timekeeping_clocksource_has_base - Check whether the current clocksource
// 				      is based on given a base clock
// @id:		base clocksource ID
//
// Note:	The return value is a snapshot which can become invalid right
// 		after the function returns.
//
// Return:	true if the timekeeper clocksource has a base clock with @id,
// 		false otherwise
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_timekeeping_clocksource_has_base_579: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32) -> bool>, *mut ()>(Some(timekeeping_clocksource_has_base)) };

/// do_settimeofday64 - Sets the time of day.
/// @ts:     pointer to the timespec64 variable containing the new time
///
/// Sets the time of day to the new time and update NTP and notify hrtimers
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.do_settimeofday64")]
extern "C" fn do_settimeofday64(ts: *const timespec64) -> i32 {
let mut tks: *mut timekeeper = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_unlock_581: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_581) as *mut *mut __SlateRecord_spinlock_h_594;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_594;
let mut xt: core::mem::MaybeUninit<timespec64> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut timespec64 = core::ptr::addr_of_mut!(xt) as *mut timespec64;
let mut ts_delta: core::mem::MaybeUninit<timespec64> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut timespec64 = core::ptr::addr_of_mut!(ts_delta) as *mut timespec64;
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
let __slate_auto_bytes = __slate_slot_3 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
if !timespec64_valid_settod(ts) {
return 22i32.wrapping_neg();
} else {
core::ptr::write(__slate_slot_2, class_raw_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(timekeeper_data[0].lock)));
__slate_cleanup_armed_2.set(true);
core::ptr::write(__slate_slot_1, (core::ptr::addr_of_mut!(timekeeper_data[0].lock) as u64) as *mut __SlateRecord_spinlock_h_594);
__slate_cleanup_armed_1.set(true);
if !class_raw_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2))).is_null() || !class_raw_spinlock_irqsave_is_conditional {
if true {
tks = core::ptr::addr_of_mut!(timekeeper_data[0].shadow_timekeeper);
timekeeping_forward_now(tks);
*__slate_slot_3 = tk_xtime(tks as *const timekeeper);
*__slate_slot_4 = timespec64_sub(*ts, *__slate_slot_3);
if timespec64_compare(core::ptr::addr_of_mut!((*tks).wall_to_monotonic) as *const timespec64, core::ptr::addr_of_mut!(*__slate_slot_4) as *const timespec64) > 0 {
timekeeping_restore_shadow((core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data).wrapping_offset(0 as isize));
return 22i32.wrapping_neg();
} else {
tk_set_wall_to_mono(tks, timespec64_sub((*tks).wall_to_monotonic, *__slate_slot_4));
tk_set_xtime(tks, ts);
timekeeping_update_from_shadow((core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data).wrapping_offset(0 as isize), 3u32);
}
}
}
__slate_cleanup_armed_1.set(false);
__class_raw_spinlock_irqsave_cleanup_ctx(__slate_slot_1 as *mut _);
__slate_cleanup_armed_2.set(false);
class_raw_spinlock_irqsave_destructor(__slate_slot_2 as *mut _);
// Signal hrtimers about time change
clock_was_set((BIT!(1i32) | BIT!(5i32) | BIT!(3i32) | BIT!(7i32)) as u32);
audit_tk_injoffset(*__slate_slot_4);
add_device_randomness(ts as *const (), core::mem::size_of::<timespec64>() as u64);
return 0;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_do_settimeofday64_582: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*const timespec64) -> i32>, *mut ()>(Some(do_settimeofday64)) };

fn timekeeper_is_core_tk(tk: *mut timekeeper) -> bool {
IS_ENABLED!() == 0 || ((unsafe { (*tk).id }) as u32) == 0
}

/// __timekeeping_inject_offset - Adds or subtracts from the current time.
/// @tkd:	Pointer to the timekeeper to modify
/// @ts:		Pointer to the timespec variable containing the offset
///
/// Adds or subtracts an offset value from the current time.
fn __timekeeping_inject_offset(tkd: *mut tk_data, ts: *const timespec64) -> i32 {
let tks: *mut timekeeper = unsafe { core::ptr::addr_of_mut!((*tkd).shadow_timekeeper) };
let mut tmp: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
if (unsafe { (*ts).tv_nsec }) < 0 || (unsafe { (*ts).tv_nsec }) >= 1000000000 {
return 22i32.wrapping_neg();
}
timekeeping_forward_now(tks);
if timekeeper_is_core_tk(tks) {
// Make sure the proposed value is valid
tmp = timespec64_add(tk_xtime(tks as *const timekeeper), unsafe { *ts });
let __v0: bool = timespec64_compare((unsafe { core::ptr::addr_of_mut!((*tks).wall_to_monotonic) }) as *const timespec64, ts) > 0 || !timespec64_valid_settod(core::ptr::addr_of_mut!(tmp) as *const timespec64);
if __v0 {
timekeeping_restore_shadow(tkd);
return 22i32.wrapping_neg();
}
tk_xtime_add(tks, ts);
tk_set_wall_to_mono(tks, timespec64_sub(unsafe { (*tks).wall_to_monotonic }, unsafe { *ts }));
} else {
let tkr_mono: *mut tk_read_base = unsafe { core::ptr::addr_of_mut!((*tks).tkr_mono) };
let mut now: i64 = 0;
let mut offs: i64 = 0;
// Get the current time
now = ktime_add_ns!(unsafe { (*tkr_mono).base }, timekeeping_get_ns(tkr_mono as *const tk_read_base)) as i64;
// Add the relative offset change
offs = ktime_add!(unsafe { (*tks).__slate_anon_6.offs_aux }, timespec64_to_ktime(unsafe { *ts }));
// Prevent that the resulting time becomes negative
if ktime_add!(now, offs) < 0 {
timekeeping_restore_shadow(tkd);
return 22i32.wrapping_neg();
}
tk_update_aux_offs(tks, offs);
}
timekeeping_update_from_shadow(tkd, 3u32);
0
}

fn timekeeping_inject_offset(ts: *const timespec64) -> i32 {
let mut __UNIQUE_ID_unlock_584: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_584) as *mut *mut __SlateRecord_spinlock_h_594;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_594;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_1 as *mut _) };
}
});
let __slate_cleanup_ptr_0 = __slate_slot_0;
let __slate_cleanup_armed_0 = core::cell::Cell::new(false);
let __slate_cleanup_0 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_0.get() {
unsafe { __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_0 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_1, class_raw_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(timekeeper_data[0].lock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(timekeeper_data[0].lock) as u64) as *mut __SlateRecord_spinlock_h_594);
__slate_cleanup_armed_0.set(true);
if !class_raw_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_raw_spinlock_irqsave_is_conditional {
if true {
*__slate_slot_2 = __timekeeping_inject_offset((core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data).wrapping_offset(0 as isize), ts);
}
}
__slate_cleanup_armed_0.set(false);
__class_raw_spinlock_irqsave_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_raw_spinlock_irqsave_destructor(__slate_slot_1 as *mut _);
// Signal hrtimers about time change
if *__slate_slot_2 == 0 {
clock_was_set((BIT!(1i32) | BIT!(5i32) | BIT!(3i32) | BIT!(7i32)) as u32);
}
return *__slate_slot_2;
}
}

/// Indicates if there is an offset between the system clock and the hardware
/// clock/persistent clock/rtc.
/// Adjust the time obtained from the CMOS to be UTC time instead of
/// local time.
///
/// This is ugly, but preferable to the alternatives.  Otherwise we
/// would either need to write a program to do it in /etc/rc (and risk
/// confusion if the program gets run more than once; it would also be
/// hard to make the program warp the clock precisely n hours)  or
/// compile in the timezone information into the kernel.  Bad, bad....
///
/// 						- TYT, 1992-01-01
///
/// The best thing to do is to keep the CMOS clock in universal time (UTC)
/// as real UNIX machines always do it. This avoids all headaches about
/// daylight saving times and warping kernel clocks.
#[unsafe(no_mangle)]
extern "C" fn timekeeping_warp_clock() {
if (unsafe { sys_tz.tz_minuteswest }) != 0 {
let mut adjust: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(adjust) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
unsafe {
persistent_clock_is_local = 1;
}
adjust.tv_sec = unsafe { sys_tz.tz_minuteswest }.wrapping_mul(60) as i64;
adjust.tv_nsec = 0;
timekeeping_inject_offset(core::ptr::addr_of_mut!(adjust) as *const timespec64);
}
}

/// __timekeeping_set_tai_offset - Sets the TAI offset from UTC and monotonic
fn __timekeeping_set_tai_offset(tk: *mut timekeeper, tai_offset: i32) {
unsafe {
(*tk).tai_offset = tai_offset;
}
unsafe {
(*tk).__slate_anon_6.offs_tai = ktime_add!(unsafe { (*tk).offs_real }, ktime_set(tai_offset as i64, 0u64));
}
}

/// change_clocksource - Swaps clocksources if a new one is available
///
/// Accumulates current time interval and initializes new clocksource
#[unsafe(link_section = ".text.slate_distinct.timekeeping.change_clocksource")]
extern "C" fn change_clocksource(data: *mut ()) -> i32 {
let mut __UNIQUE_ID_unlock_586: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_586) as *mut *mut __SlateRecord_spinlock_h_594;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_594;
// Abort if the device can't be enabled
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut old: *mut clocksource = unsafe { core::mem::zeroed() };
let mut new: *mut clocksource = unsafe { core::mem::zeroed() };
let __slate_cleanup_ptr_2 = __slate_slot_2;
let __slate_cleanup_armed_2 = core::cell::Cell::new(false);
let __slate_cleanup_2 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_2.get() {
unsafe { class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_2 as *mut _) };
}
});
let __slate_cleanup_ptr_1 = __slate_slot_1;
let __slate_cleanup_armed_1 = core::cell::Cell::new(false);
let __slate_cleanup_1 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_1.get() {
unsafe { __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _) };
}
});
unsafe {
new = data as *mut clocksource;
old = core::ptr::null_mut::<clocksource>();
// If the clocksource is in a module, get a module reference.
// Succeeds for built-in code (owner == NULL) as well. Abort if the
// reference can't be acquired.
if !try_module_get((*new).owner) {
return 0;
} else {
if (*new).enable != None {
__v3 = unsafe { (*new).enable }.unwrap()(new) != 0;
} else {
__v3 = false;
}
if __v3 {
module_put((*new).owner);
return 0;
} else {
core::ptr::write(__slate_slot_2, class_raw_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(timekeeper_data[0].lock)));
__slate_cleanup_armed_2.set(true);
core::ptr::write(__slate_slot_1, (core::ptr::addr_of_mut!(timekeeper_data[0].lock) as u64) as *mut __SlateRecord_spinlock_h_594);
__slate_cleanup_armed_1.set(true);
if !class_raw_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_2))).is_null() || !class_raw_spinlock_irqsave_is_conditional {
if true {
let tks: *mut timekeeper = core::ptr::addr_of_mut!(timekeeper_data[0].shadow_timekeeper);
timekeeping_forward_now(tks);
old = (*tks).tkr_mono.clock;
tk_setup_internals(tks, new);
timekeeping_update_from_shadow((core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data).wrapping_offset(0 as isize), 3u32);
}
}
__slate_cleanup_armed_1.set(false);
__class_raw_spinlock_irqsave_cleanup_ctx(__slate_slot_1 as *mut _);
__slate_cleanup_armed_2.set(false);
class_raw_spinlock_irqsave_destructor(__slate_slot_2 as *mut _);
tk_aux_update_clocksource();
if !old.is_null() {
if (*old).disable != None {
unsafe { (*old).disable }.unwrap()(old);
}
module_put((*old).owner);
}
return 0;
}
}
}
}

/// timekeeping_notify - Install a new clock source
/// @clock:		pointer to the clock source
///
/// This function is called from clocksource.c after a new, better clock
/// source has been registered. The caller holds the clocksource_mutex.
#[unsafe(no_mangle)]
extern "C" fn timekeeping_notify(clock: *mut clocksource) -> i32 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
if (unsafe { (*tk).tkr_mono.clock }) == clock {
return 0;
}
// Disable inlined reads accross the clocksource switch
clocksource_disable_inline_read();
unsafe { stop_machine(Some(change_clocksource), clock as *mut (), core::ptr::null::<cpumask>()) };
// If the clocksource has been selected and supports inlined reads
// enable the branch.
if (unsafe { (*tk).tkr_mono.clock }) == clock && (unsafe { (*clock).flags }) & 512 != 0 {
clocksource_enable_inline_read();
}
unsafe { tick_clock_notify() };
if (unsafe { (*tk).tkr_mono.clock }) == clock { 0 } else { 1i32.wrapping_neg() }
}

/// ktime_get_raw_ts64 - Returns the raw monotonic time in a timespec
/// @ts:		pointer to the timespec64 to be set
///
/// Returns the raw monotonic time (completely un-modified by ntp)
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_raw_ts64")]
extern "C" fn ktime_get_raw_ts64(ts: *mut timespec64) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut nsecs: u64 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
unsafe {
(*ts).tv_sec = (*tk).raw_sec as i64;
}
nsecs = timekeeping_get_ns__const3((unsafe { core::ptr::addr_of_mut!((*tk).tkr_raw) }) as *const tk_read_base);
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
unsafe {
(*ts).tv_nsec = 0;
}
timespec64_add_ns(ts, nsecs);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_raw_ts64_587: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut timespec64)>, *mut ()>(Some(ktime_get_raw_ts64)) };

/// timekeeping_valid_for_hres - Check if timekeeping is suitable for hres
#[unsafe(no_mangle)]
extern "C" fn timekeeping_valid_for_hres() -> i32 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut ret: i32 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
ret = ((unsafe { (*(*tk).tkr_mono.clock).flags }) & 32) as i32;
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
ret
}

/// timekeeping_max_deferment - Returns max time the clocksource can be deferred
#[unsafe(no_mangle)]
extern "C" fn timekeeping_max_deferment() -> u64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut ret: u64 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
ret = unsafe { (*(*tk).tkr_mono.clock).max_idle_ns };
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
ret
}

/// read_persistent_clock64 -  Return time from the persistent clock.
/// @ts: Pointer to the storage for the readout value
///
/// Weak dummy function for arches that do not yet support it.
/// Reads the time from the battery backed persistent clock.
/// Returns a timespec with tv_sec=0 and tv_nsec=0 if unsupported.
///
///  XXX - Do be sure to remove it once all arches implement it.
#[unsafe(no_mangle)]
#[linkage = "weak"]
extern "C" fn read_persistent_clock64(ts: *mut timespec64) {
unsafe {
(*ts).tv_sec = 0;
}
unsafe {
(*ts).tv_nsec = 0;
}
}

/// read_persistent_wall_and_boot_offset - Read persistent clock, and also offset
///                                        from the boot.
/// @wall_time:	  current time as returned by persistent clock
/// @boot_offset:  offset that is defined as wall_time - boot_time
///
/// Weak dummy function for arches that do not yet support it.
///
/// The default function calculates offset based on the current value of
/// local_clock(). This way architectures that support sched_clock() but don't
/// support dedicated boot time clock will provide the best estimate of the
/// boot time.
#[unsafe(no_mangle)]
#[linkage = "weak"]
#[unsafe(link_section = ".init.text")]
extern "C" fn read_persistent_wall_and_boot_offset(wall_time: *mut timespec64, boot_offset: *mut timespec64) {
read_persistent_clock64(wall_time);
unsafe {
*boot_offset = ns_to_timespec64(local_clock() as i64);
}
}

#[unsafe(link_section = ".init.text")]
fn tkd_basic_setup(tkd: *mut tk_data, __v3: u32, valid: bool) {
let mut __slate_compound_0: raw_spinlock = unsafe { core::mem::zeroed() };
'__slate_break_1: loop {
unsafe {
*core::ptr::addr_of_mut!((*tkd).lock) = *{
__slate_compound_0 = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = core::mem::zeroed();
__t0.val = __SlateRecord_types_h_190 { counter: 0 };
__t0
} } };
core::ptr::addr_of_mut!(__slate_compound_0)
};
}
if true {
break;
}
}
seqcount_raw_spinlock_init!(unsafe { core::ptr::addr_of_mut!((*tkd).seq) });
unsafe {
(*tkd).shadow_timekeeper.id = __v3;
}
unsafe {
(*tkd).timekeeper.id = __v3;
}
let __v4: u8 = valid as u8;
unsafe {
(*tkd).shadow_timekeeper.clock_valid = __v4;
}
unsafe {
(*tkd).timekeeper.clock_valid = __v4;
}
}

/// Flag reflecting whether timekeeping_resume() has injected sleeptime.
///
/// The flag starts of false and is only set when a suspend reaches
/// timekeeping_suspend(), timekeeping_resume() sets it to false when the
/// timekeeper clocksource is not stopping across suspend and has been
/// used to update sleep time. If the timekeeper clocksource has stopped
/// then the flag stays true and is used by the RTC resume code to decide
/// whether sleeptime must be injected and if so the flag gets false then.
///
/// If a suspend fails before reaching timekeeping_resume() then the flag
/// stays false and prevents erroneous sleeptime injection.
static mut suspend_timing_needed: bool = unsafe { core::mem::zeroed() };

/// Flag for if there is a persistent clock on this platform
static mut persistent_clock_exists: bool = unsafe { core::mem::zeroed() };

/// timekeeping_init - Initializes the clocksource and common timekeeping values
#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn timekeeping_init() {
let mut __slate_compound_3: timespec64 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: timespec64 = unsafe { core::mem::zeroed() };
let mut wall_time: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(wall_time) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut boot_offset: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(boot_offset) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut wall_to_mono: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(wall_to_mono) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let tks: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].shadow_timekeeper) };
tkd_basic_setup(unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize), 0u32, true);
tk_aux_setup();
read_persistent_wall_and_boot_offset(core::ptr::addr_of_mut!(wall_time), core::ptr::addr_of_mut!(boot_offset));
let __v0: bool = timespec64_valid_settod(core::ptr::addr_of_mut!(wall_time) as *const timespec64) && timespec64_to_ns(unsafe { &(*(core::ptr::addr_of_mut!(wall_time) as *const timespec64)) }) > 0;
if __v0 {
unsafe {
persistent_clock_exists = true;
}
} else {
if timespec64_to_ns(unsafe { &(*(core::ptr::addr_of_mut!(wall_time) as *const timespec64)) }) != 0 {
let mut __v1: i32;
while false {
}
__v1 = unsafe { _printk(c"\x014Persistent clock returned invalid value".as_ptr() as *const u8) };
wall_time = unsafe { *{
__slate_compound_3 = {
let mut __t0: timespec64 = core::mem::zeroed();
__t0.tv_sec = 0i64;
__t0
};
core::ptr::addr_of_mut!(__slate_compound_3)
} };
}
}
if timespec64_compare(core::ptr::addr_of_mut!(wall_time) as *const timespec64, core::ptr::addr_of_mut!(boot_offset) as *const timespec64) < 0 {
boot_offset = unsafe { *{
__slate_compound_4 = {
let mut __t1: timespec64 = core::mem::zeroed();
__t1.tv_sec = 0i64;
__t1
};
core::ptr::addr_of_mut!(__slate_compound_4)
} };
}
// We want set wall_to_mono, so the following is true:
// wall time + wall_to_mono = boot time
wall_to_mono = timespec64_sub(boot_offset, wall_time);
let clock: *mut clocksource = unsafe { clocksource_default_clock() };
if (unsafe { (*clock).enable }) != None {
unsafe { unsafe { (*clock).enable }.unwrap()(clock) };
}
let mut __UNIQUE_ID_guard_588: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) });
let __slate_cleanup_ptr_5 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_588) as *mut _;
let __slate_cleanup_5 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_5 as *mut _));
let mut __UNIQUE_ID_unlock_589: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_6 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_589) as *mut _;
let __slate_cleanup_6 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_6 as *mut _));
unsafe { ntp_init() };
tk_setup_internals(tks, clock);
tk_set_xtime(tks, core::ptr::addr_of_mut!(wall_time) as *const timespec64);
unsafe {
(*tks).raw_sec = 0;
}
tk_set_wall_to_mono(tks, wall_to_mono);
// Use TK_UPDATE_ALL so the NTP layer picks up the clocksource's
// cs_tick_adj via ntp_clear(). Clearing NTP here is otherwise
// redundant as ntp_init() already initialised it above.
timekeeping_update_from_shadow(unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize), 3u32);
}

/// time in seconds when suspend began for persistent clock
static mut timekeeping_suspend_time: timespec64 = unsafe { core::mem::zeroed() };

/// __timekeeping_inject_sleeptime - Internal function to add sleep interval
/// @tk:		Pointer to the timekeeper to be updated
/// @delta:	Pointer to the delta value in timespec64 format
///
/// Takes a timespec offset measuring a suspend interval and properly
/// adds the sleep offset to the timekeeping variables.
fn __timekeeping_inject_sleeptime(tk: *mut timekeeper, delta: *const timespec64) {
if !timespec64_valid_strict(delta) {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk_deferred(c"\x014__timekeeping_inject_sleeptime: Invalid sleep delta value!\n".as_ptr() as *const u8) };
return;
}
tk_xtime_add(tk, delta);
tk_set_wall_to_mono(tk, timespec64_sub(unsafe { (*tk).wall_to_monotonic }, unsafe { *delta }));
tk_update_sleep_time(tk, timespec64_to_ktime(unsafe { *delta }));
unsafe { tk_debug_account_sleep_time(delta) };
}

/// timekeeping_resume - Resumes the generic timekeeping subsystem.
#[unsafe(no_mangle)]
extern "C" fn timekeeping_resume() {
let tks: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].shadow_timekeeper) };
let clock: *mut clocksource = unsafe { (*tks).tkr_mono.clock };
let mut ts_new: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ts_new) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut ts_delta: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ts_delta) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut inject_sleeptime: bool = false;
let mut cycle_now: u64 = 0;
let mut nsec: u64 = 0;
let mut flags: u64 = 0;
read_persistent_clock64(core::ptr::addr_of_mut!(ts_new));
unsafe { clockevents_resume() };
unsafe { clocksource_resume() };
raw_spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }, flags);
// After system resumes, we need to calculate the suspended time and
// compensate it for the OS time. There are 3 sources that could be
// used: Nonstop clocksource during suspend, persistent clock and rtc
// device.
//
// One specific platform may have 1 or 2 or all of them, and the
// preference will be:
// 	suspend-nonstop clocksource -> persistent clock -> rtc
// The less preferred source will only be tried if there is no better
// usable source. The rtc part is handled separately in rtc core code.
cycle_now = tk_clock_read__const6((unsafe { core::ptr::addr_of_mut!((*tks).tkr_mono) }) as *const tk_read_base);
nsec = unsafe { clocksource_stop_suspend_timing(clock, cycle_now) };
if nsec > 0 {
ts_delta = unsafe { ns_to_timespec64(nsec as i64) };
inject_sleeptime = true;
} else {
if timespec64_compare(core::ptr::addr_of_mut!(ts_new) as *const timespec64, (unsafe { core::ptr::addr_of_mut!(timekeeping_suspend_time) }) as *const timespec64) > 0 {
ts_delta = timespec64_sub(ts_new, unsafe { timekeeping_suspend_time });
inject_sleeptime = true;
}
}
if inject_sleeptime {
unsafe {
suspend_timing_needed = false;
}
__timekeeping_inject_sleeptime(tks, core::ptr::addr_of_mut!(ts_delta) as *const timespec64);
}
// Re-base the last cycle value
unsafe {
(*tks).tkr_mono.cycle_last = cycle_now;
}
unsafe {
(*tks).tkr_raw.cycle_last = cycle_now;
}
unsafe {
(*tks).ntp_error = 0;
}
unsafe {
timekeeping_suspended = 0;
}
timekeeping_update_from_shadow(unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize), 2u32);
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }, flags);
touch_softlockup_watchdog();
// Resume the clockevent device(s) and hrtimers
unsafe { tick_resume() };
// Notify timerfd as resume is equivalent to clock_was_set()
unsafe { timerfd_resume() };
}

#[unsafe(link_section = ".text.slate_distinct.timekeeping.timekeeping_syscore_resume")]
extern "C" fn timekeeping_syscore_resume(data: *mut ()) {
timekeeping_resume();
}

#[unsafe(no_mangle)]
extern "C" fn timekeeping_suspend() -> i32 {
let tks: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].shadow_timekeeper) };
let mut delta: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(delta) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut delta_delta: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(delta_delta) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut flags: u64 = 0;
let mut cycle_now: u64 = 0;
read_persistent_clock64(unsafe { core::ptr::addr_of_mut!(timekeeping_suspend_time) });
// On some systems the persistent_clock can not be detected at
// timekeeping_init by its return value, so if we see a valid
// value returned, update the persistent_clock_exists flag.
if (unsafe { timekeeping_suspend_time.tv_sec }) != 0 || (unsafe { timekeeping_suspend_time.tv_nsec }) != 0 {
unsafe {
persistent_clock_exists = true;
}
}
unsafe {
suspend_timing_needed = true;
}
raw_spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }, flags);
timekeeping_forward_now(tks);
unsafe {
timekeeping_suspended = 1;
}
// Since we've called forward_now, cycle_last stores the value
// just read from the current clocksource. Save this to potentially
// use in suspend timing.
let curr_clock: *mut clocksource = unsafe { (*tks).tkr_mono.clock };
cycle_now = unsafe { (*tks).tkr_mono.cycle_last };
unsafe { clocksource_start_suspend_timing(curr_clock, cycle_now) };
if unsafe { persistent_clock_exists } {
// To avoid drift caused by repeated suspend/resumes,
// which each can add ~1 second drift error,
// try to compensate so the difference in system time
// and persistent_clock time stays close to constant.
delta = timespec64_sub(tk_xtime(tks as *const timekeeper), unsafe { timekeeping_suspend_time });
delta_delta = timespec64_sub(delta, unsafe { old_delta });
if abs!(delta_delta.tv_sec) >= 2 {
// if delta_delta is too large, assume time correction
// has occurred and set old_delta to the current delta.
unsafe {
old_delta = delta;
}
} else {
// Otherwise try to adjust old_system to compensate
unsafe {
timekeeping_suspend_time = timespec64_add(timekeeping_suspend_time, delta_delta);
}
}
}
timekeeping_update_from_shadow(unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize), 0u32);
halt_fast_timekeeper(tks as *const timekeeper);
raw_spin_unlock_irqrestore!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }, flags);
unsafe { tick_suspend() };
unsafe { clocksource_suspend() };
unsafe { clockevents_suspend() };
0
}

static mut old_delta: timespec64 = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".text.slate_distinct.timekeeping.timekeeping_syscore_suspend")]
extern "C" fn timekeeping_syscore_suspend(data: *mut ()) -> i32 {
timekeeping_suspend()
}

/// sysfs resume/suspend bits for timekeeping
static timekeeping_syscore_ops: __SlateConst<syscore_ops> = __SlateConst({
let mut __t0: syscore_ops = unsafe { core::mem::zeroed() };
__t0.suspend = Some(timekeeping_syscore_suspend);
__t0.resume = Some(timekeeping_syscore_resume);
__t0
});

static mut timekeeping_syscore: syscore = {
let mut __t0: syscore = unsafe { core::mem::zeroed() };
__t0.ops = unsafe { core::ptr::addr_of!(timekeeping_syscore_ops.0) };
__t0
};

#[unsafe(link_section = ".init.text")]
extern "C" fn timekeeping_init_ops() -> i32 {
unsafe { register_syscore(core::ptr::addr_of_mut!(timekeeping_syscore)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_timekeeping_init_ops_591: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(timekeeping_init_ops)) };

/// Apply a multiplier adjustment to the timekeeper
#[inline(always)]
fn timekeeping_apply_adjustment(tk: *mut timekeeper, mut offset: i64, mult_adj: i32) {
let mut interval: i64 = (unsafe { (*tk).cycle_interval }) as i64;
if mult_adj == 0 {
return;
} else {
if mult_adj == 1i32.wrapping_neg() {
interval = interval.wrapping_neg();
offset = offset.wrapping_neg();
} else {
if mult_adj != 1 {
interval = interval.wrapping_mul(mult_adj as i64);
offset = offset.wrapping_mul(mult_adj as i64);
}
}
}
// So the following can be confusing.
//
// To keep things simple, lets assume mult_adj == 1 for now.
//
// When mult_adj != 1, remember that the interval and offset values
// have been appropriately scaled so the math is the same.
//
// The basic idea here is that we're increasing the multiplier
// by one, this causes the xtime_interval to be incremented by
// one cycle_interval. This is because:
// 	xtime_interval = cycle_interval * mult
// So if mult is being incremented by one:
// 	xtime_interval = cycle_interval * (mult + 1)
// Its the same as:
// 	xtime_interval = (cycle_interval * mult) + cycle_interval
// Which can be shortened to:
// 	xtime_interval += cycle_interval
//
// So offset stores the non-accumulated cycles. Thus the current
// time (in shifted nanoseconds) is:
// 	now = (offset * adj) + xtime_nsec
// Now, even though we're adjusting the clock frequency, we have
// to keep time consistent. In other words, we can't jump back
// in time, and we also want to avoid jumping forward in time.
//
// So given the same offset value, we need the time to be the same
// both before and after the freq adjustment.
// 	now = (offset * adj_1) + xtime_nsec_1
// 	now = (offset * adj_2) + xtime_nsec_2
// So:
// 	(offset * adj_1) + xtime_nsec_1 =
// 		(offset * adj_2) + xtime_nsec_2
// And we know:
// 	adj_2 = adj_1 + 1
// So:
// 	(offset * adj_1) + xtime_nsec_1 =
// 		(offset * (adj_1+1)) + xtime_nsec_2
// 	(offset * adj_1) + xtime_nsec_1 =
// 		(offset * adj_1) + offset + xtime_nsec_2
// Canceling the sides:
// 	xtime_nsec_1 = offset + xtime_nsec_2
// Which gives us:
// 	xtime_nsec_2 = xtime_nsec_1 - offset
// Which simplifies to:
// 	xtime_nsec -= offset
//
// When subtracting offset from xtime_nsec, the same amount
// (in appropriate units) has to be added to ntp_error, in
// order to correctly track the delta between the time
// reported in xtime_nsec, and the intended time.
if mult_adj > 0 && unsafe { (*tk).tkr_mono.mult }.wrapping_add(mult_adj as u32) < (mult_adj as u32) {
// NTP adjustment caused clocksource mult overflow
let mut __v4: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("592: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 592b - ., 3; .popsection\n# {0}", const 592 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_timekeeping_apply_adjustment_5, sym __slate_asm_string_timekeeping_apply_adjustment_6, const 2418 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("593: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 593b - ., 4; .popsection\n# {0}", const 593 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v4 = (__ret_warn_on != 0) as i64;
return;
}
let __v10: u32 = unsafe { (*tk).tkr_mono.mult };
unsafe {
(*tk).tkr_mono.mult = __v10.wrapping_add(mult_adj as u32);
}
let __v13: u64 = unsafe { (*tk).xtime_interval };
unsafe {
(*tk).xtime_interval = __v13.wrapping_add(interval as u64);
}
let __v16: u64 = unsafe { (*tk).tkr_mono.xtime_nsec };
unsafe {
(*tk).tkr_mono.xtime_nsec = __v16.wrapping_sub(offset as u64);
}
let __v19: i64 = unsafe { (*tk).ntp_error };
unsafe {
(*tk).ntp_error = __v19.wrapping_add(offset << (*tk).ntp_error_shift);
}
}

/// Adjust the timekeeper's multiplier to the correct frequency
/// and also to reduce the accumulated error value.
fn timekeeping_adjust(tk: *mut timekeeper, offset: i64) {
let ntp_tl: u64 = unsafe { ntp_tick_length((*tk).id as u32) };
let mut skew: i64 = unsafe { ntp_get_skew_delta((*tk).id as u32) };
let mut mult: u32 = 0;
// Determine the multiplier from the current NTP tick length plus
// skew_delta. The skew biases mult so that ±1 dithering can deliver
// the time_offset slew rate. Recompute when either changes.
if likely!((unsafe { (*tk).ntp_tick }) == ntp_tl && (unsafe { (*tk).skew_delta }) == skew) != 0 {
// Revert to the base mult rate.
mult = unsafe { (*tk).tkr_mono.mult }.wrapping_sub(unsafe { (*tk).ntp_err_mult });
} else {
unsafe {
(*tk).ntp_tick = ntp_tl;
}
unsafe {
(*tk).skew_delta = skew;
}
// skew_delta is stored pre-divided by HZ (matching time_offset);
// scale it back up to the full per-tick rate for the mult bias.
skew = skew.wrapping_mul(1000);
mult = div64_u64(unsafe { (*tk).ntp_tick }.wrapping_add(skew as u64) >> unsafe { (*tk).ntp_error_shift }, unsafe { (*tk).cycle_interval }) as u32;
}
// If the clock is behind the NTP time, increase the multiplier by 1
// to catch up with it. If it's ahead and there was a remainder in the
// tick division, the clock will slow down. Otherwise it will stay
// ahead until the tick length changes to a non-divisible value.
unsafe {
(*tk).ntp_err_mult = (if (*tk).ntp_error > 0 { 1i32 } else { 0i32 }) as u32;
}
mult = mult.wrapping_add(unsafe { (*tk).ntp_err_mult });
timekeeping_apply_adjustment(tk, offset, mult.wrapping_sub(unsafe { (*tk).tkr_mono.mult }) as i32);
let __v4: bool = (unsafe { (*(*tk).tkr_mono.clock).maxadj }) != 0 && (abs__2!(unsafe { (*tk).tkr_mono.mult }.wrapping_sub(unsafe { (*(*tk).tkr_mono.clock).mult })) as u32) > unsafe { (*(*tk).tkr_mono.clock).maxadj };
if __v4 {
let mut __v7: i64;
let __ret_do_once: bool = true;
let mut __v8: i64;
let __ret_cond: bool = __ret_do_once;
let mut __ret_once: bool = false;
if unlikely__2!(__ret_cond) != 0 && unlikely__2!(!unsafe { __already_done }) != 0 {
unsafe {
__already_done = true;
}
__ret_once = true;
}
__v8 = __ret_once as i64;
if __v8 != 0 {
let mut __v9: i32;
while false {
}
__v9 = unsafe { _printk(c"\x014Adjusting %s more than 11%% (%ld vs %ld)\n".as_ptr() as *const u8, (*(*tk).tkr_mono.clock).name, (*tk).tkr_mono.mult as i64, ((*(*tk).tkr_mono.clock).mult as i64).wrapping_add((*(*tk).tkr_mono.clock).maxadj as i64)) };
}
__v7 = __ret_do_once as i64;
}
// It may be possible that when we entered this function, xtime_nsec
// was very small.  Further, if we're slightly speeding the clocksource
// in the code above, its possible the required corrective factor to
// xtime_nsec could cause it to underflow.
//
// Now, since we have already accumulated the second and the NTP
// subsystem has been notified via second_overflow(), we need to skip
// the next update.
if unlikely__2!(((unsafe { (*tk).tkr_mono.xtime_nsec }) as i64) < 0) != 0 {
let __v12: u64 = unsafe { (*tk).tkr_mono.xtime_nsec };
unsafe {
(*tk).tkr_mono.xtime_nsec = __v12.wrapping_add(1000000000u64 << (*tk).tkr_mono.shift);
}
let __v15: u64 = unsafe { (*tk).xtime_sec };
unsafe {
(*tk).xtime_sec = __v15.wrapping_sub(1);
}
unsafe {
(*tk).skip_second_overflow = 1;
}
}
}

#[unsafe(link_section = ".data..once")]
static mut __already_done: bool = unsafe { core::mem::zeroed() };

/// accumulate_nsecs_to_secs - Accumulates nsecs into secs
///
/// Helper function that accumulates the nsecs greater than a second
/// from the xtime_nsec field to the xtime_secs field.
/// It also calls into the NTP code to handle leapsecond processing.
fn accumulate_nsecs_to_secs(tk: *mut timekeeper) -> u32 {
let nsecps: u64 = 1000000000u64 << unsafe { (*tk).tkr_mono.shift };
let mut clock_set: u32 = 0;
while (unsafe { (*tk).tkr_mono.xtime_nsec }) >= nsecps {
let mut leap: i32 = 0;
let __v1: u64 = unsafe { (*tk).tkr_mono.xtime_nsec };
unsafe {
(*tk).tkr_mono.xtime_nsec = __v1.wrapping_sub(nsecps);
}
let __v4: u64 = unsafe { (*tk).xtime_sec };
unsafe {
(*tk).xtime_sec = __v4.wrapping_add(1);
}
// Skip NTP update if this second was accumulated before,
// i.e. xtime_nsec underflowed in timekeeping_adjust()
if unlikely__3!(unsafe { (*tk).skip_second_overflow }) != 0 {
unsafe {
(*tk).skip_second_overflow = 0;
}
} else {
// Figure out if its a leap sec and apply if needed
leap = unsafe { second_overflow((*tk).id as u32, (*tk).xtime_sec as i64) };
if unlikely!(leap) != 0 {
let mut ts: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ts) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let __v8: u64 = unsafe { (*tk).xtime_sec };
unsafe {
(*tk).xtime_sec = __v8.wrapping_add(leap as u64);
}
ts.tv_sec = leap as i64;
ts.tv_nsec = 0;
tk_set_wall_to_mono(tk, timespec64_sub(unsafe { (*tk).wall_to_monotonic }, ts));
__timekeeping_set_tai_offset(tk, unsafe { (*tk).tai_offset }.wrapping_sub(leap));
clock_set = 2u32;
}
}
}
clock_set
}

/// logarithmic_accumulation - shifted accumulation of cycles
///
/// This functions accumulates a shifted interval of cycles into
/// a shifted interval nanoseconds. Allows for O(log) accumulation
/// loop.
///
/// Returns the unconsumed cycles.
fn logarithmic_accumulation(tk: *mut timekeeper, mut offset: u64, shift: u32, clock_set: &mut u32) -> u64 {
let interval: u64 = (unsafe { (*tk).cycle_interval }) << shift;
let mut snsec_per_sec: u64 = 0;
// If the offset is smaller than a shifted interval, do nothing
if offset < interval {
return offset;
}
// Accumulate one shifted interval
offset = offset.wrapping_sub(interval);
let __v3: u64 = unsafe { (*tk).tkr_mono.cycle_last };
unsafe {
(*tk).tkr_mono.cycle_last = __v3.wrapping_add(interval);
}
let __v6: u64 = unsafe { (*tk).tkr_raw.cycle_last };
unsafe {
(*tk).tkr_raw.cycle_last = __v6.wrapping_add(interval);
}
let __v9: u64 = unsafe { (*tk).tkr_mono.xtime_nsec };
unsafe {
(*tk).tkr_mono.xtime_nsec = __v9.wrapping_add((*tk).xtime_interval << shift);
}
let __v11: u32 = accumulate_nsecs_to_secs(tk);
unsafe {
*clock_set = *clock_set | __v11;
}
// Accumulate raw time
let __v16: u64 = unsafe { (*tk).tkr_raw.xtime_nsec };
unsafe {
(*tk).tkr_raw.xtime_nsec = __v16.wrapping_add((*tk).raw_interval << shift);
}
snsec_per_sec = 1000000000u64 << unsafe { (*tk).tkr_raw.shift };
while (unsafe { (*tk).tkr_raw.xtime_nsec }) >= snsec_per_sec {
let __v19: u64 = unsafe { (*tk).tkr_raw.xtime_nsec };
unsafe {
(*tk).tkr_raw.xtime_nsec = __v19.wrapping_sub(snsec_per_sec);
}
let __v22: u64 = unsafe { (*tk).raw_sec };
unsafe {
(*tk).raw_sec = __v22.wrapping_add(1);
}
}
// Accumulate error between NTP and clock interval
unsafe {
(*tk).ntp_error = ((*tk).ntp_error as u64).wrapping_add((*tk).ntp_tick << shift) as i64;
}
unsafe {
(*tk).ntp_error = ((*tk).ntp_error as u64).wrapping_sub((*tk).xtime_interval << unsafe { (*tk).ntp_error_shift }.wrapping_add(shift)) as i64;
}
// When skewing, do so by adjusting ntp_error to impart an extra
// target delta into ntp_error per tick, limited to what can be
// drained from time_offset / time_adjust to avoid overshoot.
//
// The base 'mult' value was calculated with the skew taken into
// account, such that the per-tick choice of 'mult' vs. 'mult+1'
// allows for the desired effective rate and ntp_error does not
// grow unbounded.
//
// Once the full desired phase offset is delivered, any remaining
// skew imparted by the adjusted 'mult', accounted above, remains
// in ntp_error and will be compensated by the dithering over time.
if (unsafe { (*tk).skew_delta }) != 0 {
let __v31: i64 = unsafe { ntp_drain_skew((*tk).id as u32, (*tk).skew_delta << shift, shift) }.wrapping_mul(1000);
let __v33: i64 = unsafe { (*tk).ntp_error };
unsafe {
(*tk).ntp_error = __v33.wrapping_add(__v31);
}
}
offset
}

// timekeeping_advance - Updates the timekeeper to the current time and
// current NTP tick length
fn __timekeeping_advance(tkd: *mut tk_data, mode: u32) -> bool {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!((*tkd).shadow_timekeeper) };
let real_tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!((*tkd).timekeeper) };
let mut clock_set: u32 = 0;
let mut shift: i32 = 0;
let mut maxshift: i32 = 0;
let mut offset: u64 = 0;
let mut orig_offset: u64 = 0;
// Make sure we're fully resumed:
if unlikely!(unsafe { timekeeping_suspended }) != 0 {
return false;
}
offset = clocksource_delta(tk_clock_read((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base), unsafe { (*tk).tkr_mono.cycle_last }, unsafe { (*tk).tkr_mono.mask }, unsafe { (*(*tk).tkr_mono.clock).max_raw_delta });
orig_offset = offset;
// Check if there's really nothing to do
if offset < unsafe { (*real_tk).cycle_interval } && (mode as u32) == 0 {
return false;
}
// With NO_HZ we may have to accumulate many cycle_intervals
// (think "ticks") worth of time at once. To do this efficiently,
// we calculate the largest doubling multiple of cycle_intervals
// that is smaller than the offset.  We then accumulate that
// chunk in one go, and then try to consume the next smaller
// doubled multiple.
shift = ilog2__4!(offset).wrapping_sub(ilog2__4!(unsafe { (*tk).cycle_interval }));
let mut __v0: i32;
let __UNIQUE_ID_x__594: i32 = 0;
let __UNIQUE_ID_y__595: i32 = shift;
loop {
if __types_ok__4!(__UNIQUE_ID_x__594, __UNIQUE_ID_y__595) == 0 {
unsafe { __compiletime_assert_596() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__594 > __UNIQUE_ID_y__595 { __UNIQUE_ID_x__594 } else { __UNIQUE_ID_y__595 };
shift = __v0;
// Bound shift to one less than what overflows tick_length
maxshift = 64i32.wrapping_sub(ilog2__7!(unsafe { ntp_tick_length((*tk).id as u32) }).wrapping_add(1)).wrapping_sub(1);
let mut __v2: i32;
let __UNIQUE_ID_x__597: i32 = shift;
let __UNIQUE_ID_y__598: i32 = maxshift;
loop {
if __types_ok__4!(__UNIQUE_ID_x__597, __UNIQUE_ID_y__598) == 0 {
unsafe { __compiletime_assert_599() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = if __UNIQUE_ID_x__597 < __UNIQUE_ID_y__598 { __UNIQUE_ID_x__597 } else { __UNIQUE_ID_y__598 };
shift = __v2;
while offset >= unsafe { (*tk).cycle_interval } {
offset = logarithmic_accumulation(tk, offset, shift as u32, unsafe { &mut (*core::ptr::addr_of_mut!(clock_set)) });
if offset < (unsafe { (*tk).cycle_interval }) << shift {
shift = shift.wrapping_sub(1);
}
}
// Adjust the multiplier to correct NTP error
timekeeping_adjust(tk, offset as i64);
// Finally, make sure that after the rounding
// xtime_nsec isn't larger than NSEC_PER_SEC
let __v7: u32 = accumulate_nsecs_to_secs(tk);
clock_set |= __v7;
// To avoid inconsistencies caused adjtimex TK_ADV_FREQ calls
// making small negative adjustments to the base xtime_nsec
// value, only update the coarse clocks if we accumulated time
if orig_offset != offset {
tk_update_coarse_nsecs(tk);
}
timekeeping_update_from_shadow(tkd, clock_set);
clock_set != 0
}

fn timekeeping_advance(mode: u32) -> bool {
let mut __UNIQUE_ID_guard_600: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_600) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_601: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_601) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
__timekeeping_advance(unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize), mode)
}

/// update_wall_time - Uses the current clocksource to increment the wall time
///
/// It also updates the enabled auxiliary clock timekeepers
#[unsafe(no_mangle)]
extern "C" fn update_wall_time() {
if timekeeping_advance(0u32) {
unsafe { clock_was_set_delayed() };
}
tk_aux_advance();
}

/// getboottime64 - Return the real time of system boot.
/// @ts:		pointer to the timespec64 to be set
///
/// Returns the wall-time of boot in a timespec64.
///
/// This is based on the wall_to_monotonic offset and the total suspend
/// time. Calls to settimeofday will affect the value returned (which
/// basically means that however wrong your real time clock is at boot time,
/// you get the right time here).
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.getboottime64")]
extern "C" fn getboottime64(ts: *mut timespec64) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let t: i64 = ktime_sub!(unsafe { (*tk).offs_real }, unsafe { (*tk).offs_boot });
unsafe {
*ts = ktime_to_timespec64!(t);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_getboottime64_602: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut timespec64)>, *mut ()>(Some(getboottime64)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_coarse_real_ts64")]
extern "C" fn ktime_get_coarse_real_ts64(ts: *mut timespec64) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
unsafe {
*ts = tk_xtime_coarse(tk as *const timekeeper);
}
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_coarse_real_ts64_603: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut timespec64)>, *mut ()>(Some(ktime_get_coarse_real_ts64)) };

/// ktime_get_coarse_real_ts64_mg - return latter of coarse grained time or floor
/// @ts:		timespec64 to be filled
///
/// Fetch the global mg_floor value, convert it to realtime and compare it
/// to the current coarse-grained time. Fill @ts with whichever is
/// latest. Note that this is a filesystem-specific interface and should be
/// avoided outside of that context.
#[unsafe(no_mangle)]
extern "C" fn ktime_get_coarse_real_ts64_mg(ts: *mut timespec64) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let floor: u64 = atomic64_read((unsafe { core::ptr::addr_of_mut!(mg_floor.0) }) as *const __SlateRecord_types_h_197) as u64;
let mut f_real: i64 = 0;
let mut offset: i64 = 0;
let mut coarse: i64 = 0;
let mut seq: u32 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
unsafe {
*ts = tk_xtime_coarse(tk as *const timekeeper);
}
offset = unsafe { timekeeper_data[0].timekeeper.offs_real };
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
coarse = timespec64_to_ktime(unsafe { *ts });
f_real = ktime_add__2!(floor, offset) as i64;
if ktime_after(f_real, coarse) {
unsafe {
*ts = ktime_to_timespec64!(f_real);
}
}
}

/// ktime_get_real_ts64_mg - attempt to update floor value and return result
/// @ts:		pointer to the timespec to be set
///
/// Get a monotonic fine-grained time value and attempt to swap it into
/// mg_floor. If that succeeds then accept the new floor value. If it fails
/// then another task raced in during the interim time and updated the
/// floor.  Since any update to the floor must be later than the previous
/// floor, either outcome is acceptable.
///
/// Typically this will be called after calling ktime_get_coarse_real_ts64_mg(),
/// and determining that the resulting coarse-grained timestamp did not effect
/// a change in ctime. Any more recent floor value would effect a change to
/// ctime, so there is no need to retry the atomic64_try_cmpxchg() on failure.
///
/// @ts will be filled with the latest floor value, regardless of the outcome of
/// the cmpxchg. Note that this is a filesystem specific interface and should be
/// avoided outside of that context.
#[unsafe(no_mangle)]
extern "C" fn ktime_get_real_ts64_mg(ts: *mut timespec64) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut old: i64 = atomic64_read((unsafe { core::ptr::addr_of_mut!(mg_floor.0) }) as *const __SlateRecord_types_h_197);
let mut offset: i64 = 0;
let mut mono: i64 = 0;
let mut seq: u32 = 0;
let mut nsecs: u64 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
unsafe {
(*ts).tv_sec = (*tk).xtime_sec as i64;
}
mono = unsafe { (*tk).tkr_mono.base };
nsecs = timekeeping_get_ns__const5((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base);
offset = unsafe { timekeeper_data[0].timekeeper.offs_real };
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
mono = ktime_add_ns!(mono, nsecs) as i64;
// Attempt to update the floor with the new time value. As any
// update must be later then the existing floor, and would effect
// a change to ctime from the perspective of the current task,
// accept the resulting floor value regardless of the outcome of
// the swap.
if atomic64_try_cmpxchg(unsafe { core::ptr::addr_of_mut!(mg_floor.0) }, core::ptr::addr_of_mut!(old), mono) {
unsafe {
(*ts).tv_nsec = 0;
}
timespec64_add_ns(ts, nsecs);
timekeeping_inc_mg_floor_swaps();
} else {
// Another task changed mg_floor since "old" was fetched.
// "old" has been updated with the latest value of "mg_floor".
// That value is newer than the previous floor value, which
// is enough to effect a change to ctime. Accept it.
unsafe {
*ts = ktime_to_timespec64!(ktime_add!(old, offset));
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.timekeeping.ktime_get_coarse_ts64")]
extern "C" fn ktime_get_coarse_ts64(ts: *mut timespec64) {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut now: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(now) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut mono: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(mono) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut seq: u32 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
now = tk_xtime_coarse(tk as *const timekeeper);
mono = unsafe { (*tk).wall_to_monotonic };
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
unsafe { set_normalized_timespec64(ts, now.tv_sec.wrapping_add(mono.tv_sec), now.tv_nsec.wrapping_add(mono.tv_nsec)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ktime_get_coarse_ts64_604: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut timespec64)>, *mut ()>(Some(ktime_get_coarse_ts64)) };

/// Must hold jiffies_lock
#[unsafe(no_mangle)]
extern "C" fn do_timer(ticks: u64) {
let __v0: u64 = unsafe { jiffies_64 };
unsafe {
jiffies_64 = __v0.wrapping_add(ticks);
}
unsafe { calc_global_load() };
}

/// ktime_get_update_offsets_now - hrtimer helper
/// @cwsseq:	pointer to check and store the clock was set sequence number
/// @offs_real:	pointer to storage for monotonic -> realtime offset
/// @offs_boot:	pointer to storage for monotonic -> boottime offset
/// @offs_tai:	pointer to storage for monotonic -> clock tai offset
///
/// Returns current monotonic time and updates the offsets if the
/// sequence number in @cwsseq and timekeeper.clock_was_set_seq are
/// different.
///
/// Called from hrtimer_interrupt() or retrigger_next_event()
#[unsafe(no_mangle)]
extern "C" fn ktime_get_update_offsets_now(cwsseq: *mut u32, offs_real: *mut i64, offs_boot: *mut i64, offs_tai: *mut i64) -> i64 {
let tk: *mut timekeeper = unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].timekeeper) };
let mut seq: u32 = 0;
let mut base: i64 = 0;
let mut nsecs: u64 = 0;
loop {
seq = read_seqcount_begin!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) });
base = unsafe { (*tk).tkr_mono.base };
nsecs = timekeeping_get_ns__const5((unsafe { core::ptr::addr_of_mut!((*tk).tkr_mono) }) as *const tk_read_base);
base = ktime_add_ns!(base, nsecs) as i64;
if (unsafe { *cwsseq }) != unsafe { (*tk).clock_was_set_seq } {
unsafe {
*cwsseq = (*tk).clock_was_set_seq;
}
unsafe {
*offs_real = (*tk).offs_real;
}
unsafe {
*offs_boot = (*tk).offs_boot;
}
unsafe {
*offs_tai = (*tk).__slate_anon_6.offs_tai;
}
}
// Handle leapsecond insertion adjustments
if unlikely__2!(base >= unsafe { (*tk).next_leap_ktime }) != 0 {
unsafe {
*offs_real = ktime_sub!(unsafe { (*tk).offs_real }, ktime_set(1i64, 0u64));
}
}
if read_seqcount_retry!(unsafe { core::ptr::addr_of_mut!(timekeeper_data[0].seq) }, seq) == 0 {
break;
}
}
base
}

/// timekeeping_validate_timex - Ensures the timex is ok for use in do_adjtimex
fn timekeeping_validate_timex(txc: *const __kernel_timex, aux_clock: bool) -> i32 {
if (unsafe { (*txc).modes }) & 32768 != 0 {
// singleshot must not be used with any other mode bits
if (unsafe { (*txc).modes }) & 1 == 0 {
return 22i32.wrapping_neg();
}
let __v0: bool = (unsafe { (*txc).modes }) & 8192 == 0 && !unsafe { capable(25i32) };
if __v0 {
return 1i32.wrapping_neg();
}
} else {
// In order to modify anything, you gotta be super-user!
let __v1: bool = (unsafe { (*txc).modes }) != 0 && !unsafe { capable(25i32) };
if __v1 {
return 1i32.wrapping_neg();
}
// if the quartz is off by more than 10% then
// something is VERY wrong!
if (unsafe { (*txc).modes }) & 16384 != 0 && ((unsafe { (*txc).tick }) < 9000i64 || (unsafe { (*txc).tick }) > 11000i64) {
return 22i32.wrapping_neg();
}
}
if (unsafe { (*txc).modes }) & 256 != 0 {
// In order to inject time, you gotta be super-user!
if !unsafe { capable(25i32) } {
return 1i32.wrapping_neg();
}
// Validate if a timespec/timeval used to inject a time
// offset is valid.  Offsets can be positive or negative, so
// we don't check tv_sec. The value of the timeval/timespec
// is the sum of its fields,but *NOTE*:
// The field tv_usec/tv_nsec must always be non-negative and
// we can't have more nanoseconds/microseconds than a second.
if (unsafe { (*txc).time.tv_usec }) < 0 {
return 22i32.wrapping_neg();
}
if (unsafe { (*txc).modes }) & 8192 != 0 {
if (unsafe { (*txc).time.tv_usec }) >= 1000000000 {
return 22i32.wrapping_neg();
}
} else {
if (unsafe { (*txc).time.tv_usec }) >= 1000000 {
return 22i32.wrapping_neg();
}
}
}
// Check for potential multiplication overflows that can
// only happen on 64-bit systems:
if (unsafe { (*txc).modes }) & 2 != 0 && true {
if 9223372036854775807i64.wrapping_neg().wrapping_sub(1) / 65536000i64 > unsafe { (*txc).freq } {
return 22i32.wrapping_neg();
}
if 140737488355i64 < unsafe { (*txc).freq } {
return 22i32.wrapping_neg();
}
}
if aux_clock {
// Auxiliary clocks are similar to TAI and do not have leap seconds
if (unsafe { (*txc).modes }) & 16 != 0 && (unsafe { (*txc).status }) & (16 | 32) != 0 {
return 22i32.wrapping_neg();
}
// No TAI offset setting
if (unsafe { (*txc).modes }) & 128 != 0 {
return 22i32.wrapping_neg();
}
// No PPS support either
if (unsafe { (*txc).modes }) & 16 != 0 && (unsafe { (*txc).status }) & (2 | 4) != 0 {
return 22i32.wrapping_neg();
}
}
0
}

// random_get_entropy_fallback - Returns the raw clock source value,
// used by random.c for platforms with no valid random_get_entropy().
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_random_get_entropy_fallback_606: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> u64>, *mut ()>(Some(random_get_entropy_fallback)) };

#[repr(C)]
#[derive(Clone, Copy)]
struct adjtimex_result {
    ad: audit_ntp_data,
    delta: timespec64,
    clock_set: bool,
}


fn __do_adjtimex(tkd: *mut tk_data, txc: *mut __kernel_timex, result: &mut adjtimex_result) -> i32 {
let tks: *mut timekeeper = unsafe { core::ptr::addr_of_mut!((*tkd).shadow_timekeeper) };
let aux_clock: bool = !timekeeper_is_core_tk(tks);
let mut ts: timespec64 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(ts) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 16usize);
}
let mut orig_tai: i32 = 0;
let mut tai: i32 = 0;
let mut ret: i32 = 0;
// Validate the data before disabling interrupts
ret = timekeeping_validate_timex(txc as *const __kernel_timex, aux_clock);
if ret != 0 {
return ret;
}
unsafe { add_device_randomness(txc as *const (), core::mem::size_of::<__kernel_timex>() as u64) };
if !aux_clock {
ktime_get_real_ts64(core::ptr::addr_of_mut!(ts));
} else {
if !tk_get_aux_ts64((unsafe { (*tkd).timekeeper.id }) as u32, unsafe { &(*core::ptr::addr_of_mut!(ts)) }) {
return 19i32.wrapping_neg();
}
}
unsafe { add_device_randomness(core::ptr::addr_of_mut!(ts) as *const (), core::mem::size_of::<timespec64>() as u64) };
let mut __UNIQUE_ID_guard_607: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!((*tkd).lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_607) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_608: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!((*tkd).lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_608) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if (unsafe { (*tks).clock_valid }) == 0 {
return 19i32.wrapping_neg();
}
if (unsafe { (*txc).modes }) & 256 != 0 {
unsafe {
(*(result as *mut adjtimex_result)).delta.tv_sec = (*txc).time.tv_sec;
}
unsafe {
(*(result as *mut adjtimex_result)).delta.tv_nsec = (*txc).time.tv_usec;
}
if (unsafe { (*txc).modes }) & 8192 == 0 {
let __v3: i64 = unsafe { (*result).delta.tv_nsec };
unsafe {
(*result).delta.tv_nsec = __v3.wrapping_mul(1000);
}
}
ret = __timekeeping_inject_offset(tkd, (unsafe { core::ptr::addr_of_mut!((*(result as *mut adjtimex_result)).delta) }) as *const timespec64);
if ret != 0 {
return ret;
}
unsafe {
(*(result as *mut adjtimex_result)).clock_set = true;
}
}
let __v5: i32 = unsafe { (*tks).tai_offset };
tai = __v5;
orig_tai = __v5;
ret = unsafe { ntp_adjtimex((*tks).id as u32, txc, core::ptr::addr_of_mut!(ts) as *const timespec64, core::ptr::addr_of_mut!(tai), core::ptr::addr_of_mut!((*(result as *mut adjtimex_result)).ad)) };
if tai != orig_tai {
__timekeeping_set_tai_offset(tks, tai);
timekeeping_update_from_shadow(tkd, 2u32);
unsafe {
(*(result as *mut adjtimex_result)).clock_set = true;
}
} else {
tk_update_leap_state_all(tkd);
}
// Update the multiplier immediately if frequency was set directly
if (unsafe { (*txc).modes }) & 16386u32 != 0 {
let __v6: bool = __timekeeping_advance(tkd, 1u32);
unsafe {
(*result).clock_set = ((*result).clock_set as i32) | (__v6 as i32) != 0;
}
}
ret
}

/// do_adjtimex() - Accessor function to NTP __do_adjtimex function
/// @txc:	Pointer to kernel_timex structure containing NTP parameters
#[unsafe(no_mangle)]
extern "C" fn do_adjtimex(txc: *mut __kernel_timex) -> i32 {
let mut result: adjtimex_result = {
let mut __t0: adjtimex_result = unsafe { core::mem::zeroed() };
__t0
};
let ret: i32 = __do_adjtimex(unsafe { core::ptr::addr_of_mut!(timekeeper_data) as *mut tk_data }.wrapping_offset(0 as isize), txc, unsafe { &mut (*core::ptr::addr_of_mut!(result)) });
if ret < 0 {
return ret;
}
if (unsafe { (*txc).modes }) & 256 != 0 {
audit_tk_injoffset(result.delta);
}
audit_ntp_log(core::ptr::addr_of_mut!(result.ad) as *const audit_ntp_data);
if result.clock_set {
unsafe { clock_was_set((BIT!(1i32) | BIT!(5i32) | BIT!(3i32) | BIT!(7i32)) as u32) };
}
unsafe { ntp_notify_cmos_timer(result.delta.tv_sec != 0) };
ret
}

/// Invoked from NTP with the time keeper lock held, so lockless access is
/// fine.
#[unsafe(no_mangle)]
extern "C" fn ktime_get_ntp_seconds(id: u32) -> i64 {
(unsafe { timekeeper_data[id as usize].timekeeper.xtime_sec }) as i64
}

static __slate_asm_string_get_device_system_crosststamp_2: [u8; 1] = *b"\0";

static __slate_asm_string_get_device_system_crosststamp_3: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_ktime_get_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_ktime_get_coarse_with_offset_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_coarse_with_offset_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_ktime_get_real_ts64_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_real_ts64_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_ktime_get_resolution_ns_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_resolution_ns_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_ktime_get_seconds_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_seconds_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_ktime_get_snapshot_id_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_snapshot_id_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_ktime_get_snapshot_id_7: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_snapshot_id_8: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_ktime_get_ts64_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_ts64_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_ktime_get_with_offset_1: [u8; 1] = *b"\0";

static __slate_asm_string_ktime_get_with_offset_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_timekeeping_apply_adjustment_5: [u8; 1] = *b"\0";

static __slate_asm_string_timekeeping_apply_adjustment_6: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

static __slate_asm_string_tk_set_wall_to_mono_1: [u8; 1] = *b"\0";

static __slate_asm_string_tk_set_wall_to_mono_2: [u8; 55] = *b"/home/takashi/c-corpus/linux/kernel/time/timekeeping.c\0";

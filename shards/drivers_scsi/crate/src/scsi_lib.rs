
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
//! Copyright (C) 1999 Eric Youngdale
//! Copyright (C) 2014 Christoph Hellwig
//!
//!  SCSI queueing library.
//!      Initial versions: Eric Youngdale (eric@andante.org).
//!                        Based upon conversations with large numbers
//!                        of people at Linux Expo.
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
//! Sign Flag
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
//! Prevent speculative execution past this barrier.
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
//! lower_32_bits - return bits 0-31 of a number
//! @n: the number we're accessing
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
//! The IDR API does not expose the tagging functionality of the radix tree
//! to users.  Use tag 0 to track whether a node has free space below it.
//! number of env pointers
//! Need to take variadic arguments even though we don't use them, as dev_fmt()
//! may only just have been expanded and may result in multiple arguments.
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
//! iterate over multi-page bvec
//! Maximum number of blkcg policies allowed to be registered concurrently.
//! Defined here to simplify include dependency.
//! See the comment in asm-generic/io.h about ioremap_np().
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! The microseconds/nanosecond delay multiplicators are used to convert a
//! constant microseconds/nanoseconds value to a value which can be used by the
//! architectures specific implementation to transform it into loops.
//! DIX-capable adapters effectively support infinite chaining for the
//! protection information scatterlist
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
// Overflow Flag
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
// informational
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
// min_not_zero - return the minimum that is _not_ zero, unless both are zero
// @x: value1
// @y: value2
// Use these carefully: no type checking, and uses the arguments
// multiple times. Use for obvious constants only.
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
// N 32-bit words worth of info
// signal notifications exist
// rescheduling necessary
// idle is polling for TIF_NEED_RESCHED
// re-arm the timer
// disable sas during sighandling
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
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
// PCI/ISA I/O ports
// Resource extended types
// Userland may not map this resource
// No address assigned yet
// Set the IDR flag and the IDR_FREE tag
// Some callsites directly call dev_printk rather than going through the
// dev_<level> infrastructure, so we need to emit here as well as inside those
// level-specific macros. Only one index entry will be produced, either way,
// since dev_printk's `fmt` isn't known at compile time if going through the
// dev_<level> macros.
//
// dev_fmt() isn't called for dev_printk when used directly, as it's used by
// the dev_<level> macros internally which already have dev_fmt() processed.
//
// We also can't use dev_printk_index_wrap directly, because we have a separate
// level to process.
// Generate SD flag bits
// Implementation specific max size
// Device/Port type
// Root Port
// Link Control
// Root Control
// Link Control 2
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
// Error values that may be returned by PCI functions
// Allow INTx interrupts
// Allow MSI interrupts
// Allow MSI-X interrupts
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
// The maximum constant udelay/ndelay value picked out of thin air to prevent
// too long constant udelays/ndelays.
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

macro_rules! SB_NR_TO_BIT {
($sb:expr, $bitnr:expr) => {
($bitnr) & (1u32 << unsafe { (*$sb).shift }).wrapping_sub(1)
};
}

macro_rules! SB_NR_TO_INDEX {
($sb:expr, $bitnr:expr) => {
($bitnr) >> unsafe { (*$sb).shift }
};
}

macro_rules! SCSI_CHECK_LOGGING {
() => {
while false {
}
};
}

/// These are the macros that are actually used throughout the code to
/// log events.  If logging isn't enabled, they are no-ops and will be
/// completely absent from the user's code.
macro_rules! SCSI_LOG_ERROR_RECOVERY {
() => {
SCSI_CHECK_LOGGING!();
};
}

macro_rules! SCSI_LOG_HLCOMPLETE {
() => {
SCSI_CHECK_LOGGING!();
};
}

macro_rules! SCSI_LOG_LEVEL {
() => {
0i32
};
}

macro_rules! SCSI_LOG_MLQUEUE {
() => {
SCSI_CHECK_LOGGING!();
};
}

macro_rules! UL {
() => {
_UL!()
};
}

macro_rules! VM_WARN_ON_ONCE {
() => {
BUILD_BUG_ON_INVALID!();
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

macro_rules! _BITUL {
($x:expr) => {
_UL!() << $x
};
}

macro_rules! _UL {
() => {
_AC!()
};
}

macro_rules! __READ_ONCE {
($x:expr) => {
unsafe { core::ptr::read_volatile(core::ptr::addr_of!($x) as *const i32) }
};
}

macro_rules! __SLAB_FLAG_BIT {
($nr:expr) => {
1u32 << $nr
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

macro_rules! __WRITE_ONCE__7 {
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
($x:expr) => {
__acquire_ctx_lock__2(($x) as *const spinlock);
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

macro_rules! __clear_bit {
($nr:expr, $addr:expr) => {
if true && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
generic___clear_bit(($nr) as u64, ($addr) as *mut u64);
} else {
___clear_bit(($nr) as u64, ($addr) as *mut u64);
}
};
}

macro_rules! __cmp_once__5 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = ($y) as $type;
__v = if __v2 < __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once__6 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: i32;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if (__v2 as i32) < (__v3 as i32) { __v2 as i32 } else { __v3 as i32 };
__v
}
};
}

macro_rules! __cmp_once__4 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = ($x) as $type;
let __v3: $type = $y;
__v = if __v2 > __v3 { __v2 } else { __v3 };
__v
}
};
}

macro_rules! __cmp_once__7 {
($type:ty, $x:expr, $y:expr) => {
{
let mut __v: $type;
let __v2: $type = $x;
let __v3: $type = ($y) as $type;
__v = if __v2 > __v3 { __v2 } else { __v3 };
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

macro_rules! __ismask {
($x:expr) => {
unsafe { *(core::ptr::addr_of!(_ctype) as *const u8).wrapping_offset(((($x) as u8) as i32) as isize) }
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
macro_rules! __native_word {
() => {
((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__8 {
() => {
((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__5 {
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

macro_rules! __ratelimit__2 {
($state:expr) => {
unsafe { ___ratelimit($state, c"scsi_io_completion_action".as_ptr() as *const u8) }
};
}

/// __release() - function to release context lock exclusively
/// @x: context lock instance pointer
///
/// No-op function that releases the given context lock instance @x.
macro_rules! __release {
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

/// Are two types/vars the same type (ignoring qualifiers)?
macro_rules! __same_type {
() => {
0i32
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
if is_signed_type!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul(((core::mem::size_of::<i32>() as u64) < 4) as i32)) }
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
macro_rules! __sign_use__4 {
($ux:expr) => {
if is_signed_type__4!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u16>() }) as u64) < 4) as i32)) }
};
}

macro_rules! __smp_mb__after_atomic {
() => {
while false {
}
};
}

macro_rules! __swab16 {
($x:expr) => {
unsafe { __slate_intrinsic_1a0c079c71348b00(($x) as u16) }
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

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use__4!($ux) & __sign_use__4!($uy)
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
macro_rules! __verify_pcpu_ptr__6 {
() => {
loop {
let __vpp_verify: *const () = core::ptr::null_mut::<u32>() as *const ();
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
macro_rules! __verify_pcpu_ptr__5 {
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
macro_rules! __verify_pcpu_ptr__3 {
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

macro_rules! absolute_pointer {
($val:expr) => {
RELOC_HIDE!(($val) as *mut (), 0i32)
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

macro_rules! blk_queue_dying {
($q:expr) => {
test_bit!(0i32, unsafe { core::ptr::addr_of_mut!((*$q).queue_flags) })
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
macro_rules! container_of__3 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__7!($type) as isize)) }) as *mut ()) as *mut $type;
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
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__21!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__10 {
($ptr:expr) => {
{
let mut __v: *mut Scsi_Host;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__32!() as isize)) }) as *mut ()) as *mut Scsi_Host;
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
macro_rules! container_of__20 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__22!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__22 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__29!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of__24 {
($ptr:expr, $type:ty) => {
{
let mut __v: *mut $type;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__14!() as isize)) }) as *mut ()) as *mut $type;
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
macro_rules! container_of {
($ptr:expr) => {
{
let mut __v: *mut scsi_device;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__3!() as isize)) }) as *mut ()) as *mut scsi_device;
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

macro_rules! host_byte {
($result:expr) => {
($result) >> 16i32 & 255
};
}

macro_rules! irqs_disabled {
() => {
{
let mut __v: i32;
let mut __v2: u64 = 0;
raw_local_save_flags!(__v2);
let mut __v3: i32;
typecheck__5!(u64);
__v3 = arch_irqs_disabled_flags(__v2);
__v = __v3;
__v
}
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type__4 {
() => {
((1i32.wrapping_neg() as u16) as i32) < 1
};
}

/// Whether 'type' is a signed type or an unsigned type. Supports scalar types,
/// bool and also pointer types.
macro_rules! is_signed_type {
() => {
1i32.wrapping_neg() < 1
};
}

/// Note: isspace() must return false for %NUL-terminator
macro_rules! isspace {
($c:expr) => {
(__ismask!($c) as i32) & 32 != 0
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

/// kcsan_check_write - check regular write access for races
///
/// @ptr: address of access
/// @size: size of access
macro_rules! kcsan_check_write {
($ptr:expr, $size:expr) => {
kcsan_check_access(absolute_pointer!($ptr) as *const (), $size, 1i32);
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

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__2 {
($ptr:expr, $type:ty) => {
container_of__3!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__15 {
($ptr:expr, $type:ty) => {
container_of__22!($ptr, $type)
};
}

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry {
($ptr:expr) => {
container_of!($ptr)
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
macro_rules! list_first_entry__2 {
($ptr:expr) => {
list_entry!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__2 {
($pos:expr, $member:ident) => {
list_entry!(unsafe { (*$pos).$member.next })
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__4!($type, $x, $y)
};
}

/// max_t - return maximum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! max_t__2 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__7!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__4 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__6!($type, $x, $y)
};
}

/// min_t - return minimum of two values, using the specified type
/// @type: data type to use
/// @x: first value
/// @y: second value
macro_rules! min_t__5 {
($type:ty, $x:expr, $y:expr) => {
__cmp_once__5!($type, $x, $y)
};
}

macro_rules! offset_in_page {
($p:expr) => {
(($p) as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))
};
}

macro_rules! offsetof__14 {
() => {
400u64
};
}

macro_rules! offsetof__4 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, gendrv) as u64
};
}

macro_rules! offsetof {
() => {
40u64
};
}

macro_rules! offsetof__21 {
() => {
216u64
};
}

macro_rules! offsetof__7 {
($TYPE:ty) => {
core::mem::offset_of!($TYPE, node) as u64
};
}

macro_rules! offsetof__22 {
() => {
1976u64
};
}

macro_rules! offsetof__29 {
() => {
96u64
};
}

macro_rules! offsetof__32 {
() => {
176u64
};
}

macro_rules! offsetof__3 {
() => {
16u64
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

macro_rules! raw_local_save_flags {
($flags:expr) => {
loop {
typecheck__5!(u64);
$flags = arch_local_save_flags();
if true {
break;
}
}
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

macro_rules! rq_dma_dir {
($rq:expr) => {
if op_is_write(req_op(($rq) as *const request) as u32) { 1i32 } else { 2i32 }
};
}

macro_rules! smp_mb__after_atomic {
() => {
loop {
kcsan_mb!();
__smp_mb__after_atomic!();
if true {
break;
}
}
};
}

macro_rules! spin_lock_irqsave {
($lock:expr, $flags:expr) => {
loop {
raw_spin_lock_irqsave!(spinlock_check($lock), $flags);
__release!(spinlock_check($lock));
__acquire!($lock);
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

macro_rules! static_branch_unlikely__2 {
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

macro_rules! static_branch_unlikely__3 {
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
let __v: bool = typecheck__4!(u64) != 0 && typecheck__3!(u64) != 0;
__v && ($b.wrapping_sub($a) as i64) < 0
}
};
}

/// time_before - returns true if the time a is before time b.
/// @a: first comparable as unsigned long
/// @b: second comparable as unsigned long
///
/// Return: %true is time a is before time b, otherwise %false.
macro_rules! time_before__2 {
($a:expr, $b:expr) => {
time_after__2!($b, $a)
};
}

macro_rules! to_scsi_driver {
($drv:expr) => {
{
let mut __v: *mut scsi_driver;
__v = ((unsafe { ((($drv) as *mut ()) as *const u8).wrapping_offset(-(offsetof__4!(scsi_driver) as isize)) }) as *mut ()) as *mut scsi_driver;
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
macro_rules! typecheck__5 {
($type:ty) => {
let mut __v: i32;
let mut __v2: $type = 0;
let mut __v3: $type = 0;
core::ptr::addr_of_mut!(__v2) == core::ptr::addr_of_mut!(__v3);
__v = 1;
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

macro_rules! unlikely__4 {
($x:expr) => {
(($x) != 0) as i64
};
}

/// virt_to_page(kaddr) returns a valid pointer if and only if
/// virt_addr_valid(kaddr) returns true.
macro_rules! virt_to_page {
($kaddr:expr) => {
((unsafe { vmemmap_base }) as *mut page).wrapping_offset((__pa!($kaddr) >> 12i32) as isize)
};
}

unsafe extern "C" {
    static mut this_cpu_off: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut vmemmap_base: u64;
    static mut __cpu_online_mask: cpumask;
    static mut cpuinfo_op: seq_operations;
    static mut __preempt_count: u64;
    static mut __SCK__preempt_schedule: static_call_key;
    static mut cpu_number: i32;
    static mut runqueues: rq;
    static mut RCU: *const __ctx_lock_RCU;
    static mut jiffies: u64;
    static mut system_percpu_wq: *mut workqueue_struct;
    static mut numa_node: i32;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut _ctype: [u8; 0];
    static mut scsi_command_size_tbl: [u8; 8];
    static mut tracepoint_srcu: srcu_struct;
    static mut __SCK__tp_func_scsi_dispatch_cmd_start: static_call_key;
    static mut __tracepoint_scsi_dispatch_cmd_start: tracepoint;
    static mut __SCK__tp_func_scsi_dispatch_cmd_error: static_call_key;
    static mut __tracepoint_scsi_dispatch_cmd_error: tracepoint;
    static mut __SCK__tp_func_scsi_dispatch_cmd_done: static_call_key;
    static mut __tracepoint_scsi_dispatch_cmd_done: tracepoint;
    static mut scsi_logging_level: u32;
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn __bad_size_call_parameter();
    fn __compiletime_assert_0();
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn ___ratelimit(arg0: *mut ratelimit_state, arg1: *const u8) -> i32;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn ____wrong_branch_error() -> bool;
    fn __cmpxchg_wrong_size();
    fn __xadd_wrong_size();
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __compiletime_assert_14();
    fn __compiletime_assert_15();
    fn __compiletime_assert_16();
    fn __compiletime_assert_19();
    fn __compiletime_assert_20();
    fn _raw_spin_lock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock_irq(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn delayed_work_timer_fn(arg0: *mut timer_list);
    fn ___migrate_enable();
    fn complete(arg0: *mut completion);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn synchronize_rcu();
    fn __rcu_read_lock();
    fn __rcu_read_unlock();
    fn rcu_is_watching() -> bool;
    fn timer_init_key(arg0: *mut timer_list, arg1: Option<unsafe extern "C" fn(*mut timer_list)>, arg2: u32, arg3: *const u8, arg4: *mut lock_class_key);
    fn queue_work_on(arg0: i32, arg1: *mut workqueue_struct, arg2: *mut work_struct) -> bool;
    fn __compiletime_assert_168();
    fn __srcu_check_read_flavor(arg0: *mut srcu_struct, arg1: i32);
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_360();
    fn __compiletime_assert_385();
    fn __compiletime_assert_386();
    fn __kmem_cache_create_args(arg0: *const u8, arg1: u32, arg2: *mut kmem_cache_args, arg3: u32) -> *mut kmem_cache;
    fn kmem_cache_destroy(arg0: *mut kmem_cache);
    fn kfree(arg0: *const ());
    fn __compiletime_assert_391();
    fn __builtin_unreachable();
    fn kmem_cache_free(arg0: *mut kmem_cache, arg1: *mut ());
    fn kmem_cache_alloc_node_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: i32) -> *mut ();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn kobject_uevent_env(arg0: *mut kobject, arg1: u32, arg2: *mut *mut u8) -> i32;
    fn device_for_each_child(arg0: *mut device, arg1: *mut (), arg2: Option<unsafe extern "C" fn(*mut device, *mut ()) -> i32>) -> i32;
    fn get_device(arg0: *mut device) -> *mut device;
    fn put_device(arg0: *mut device);
    fn sbitmap_get(arg0: *mut sbitmap) -> i32;
    fn sbitmap_weight(arg0: *const sbitmap) -> u32;
    fn blk_set_pm_only(arg0: *mut request_queue);
    fn blk_clear_pm_only(arg0: *mut request_queue);
    fn add_disk_randomness(arg0: *mut gendisk);
    fn blk_get_queue(arg0: *mut request_queue) -> bool;
    fn blk_put_queue(arg0: *mut request_queue);
    fn kblockd_schedule_work(arg0: *mut work_struct) -> i32;
    fn sg_zero_buffer(arg0: *mut scatterlist, arg1: u32, arg2: u64, arg3: i64) -> u64;
    fn sg_free_table_chained(arg0: *mut sg_table, arg1: u32);
    fn sg_alloc_table_chained(arg0: *mut sg_table, arg1: i32, arg2: *mut scatterlist, arg3: u32) -> i32;
    fn blk_mq_alloc_tag_set(arg0: *mut blk_mq_tag_set) -> i32;
    fn blk_mq_free_tag_set(arg0: *mut blk_mq_tag_set);
    fn blk_mq_free_request(arg0: *mut request);
    fn blk_mq_alloc_request(arg0: *mut request_queue, arg1: u32, arg2: u32) -> *mut request;
    fn __compiletime_assert_585();
    fn blk_mq_start_request(arg0: *mut request);
    fn __blk_mq_end_request(arg0: *mut request, arg1: u8);
    fn blk_mq_requeue_request(arg0: *mut request, arg1: bool);
    fn blk_mq_kick_requeue_list(arg0: *mut request_queue);
    fn blk_mq_delay_kick_requeue_list(arg0: *mut request_queue, arg1: u64);
    fn blk_mq_complete_request(arg0: *mut request);
    fn blk_mq_wait_quiesce_done(arg0: *mut blk_mq_tag_set);
    fn blk_mq_unquiesce_queue(arg0: *mut request_queue);
    fn blk_mq_run_hw_queues(arg0: *mut request_queue, arg1: bool);
    fn blk_mq_delay_run_hw_queues(arg0: *mut request_queue, arg1: u64);
    fn blk_mq_freeze_queue_nomemsave(arg0: *mut request_queue);
    fn blk_mq_unfreeze_queue_nomemrestore(arg0: *mut request_queue);
    fn blk_mq_map_queues(arg0: *mut blk_mq_queue_map);
    fn blk_mq_quiesce_queue_nowait(arg0: *mut request_queue);
    fn __blk_should_fake_timeout(arg0: *mut request_queue) -> bool;
    fn blk_rq_map_kern(arg0: *mut request, arg1: *mut (), arg2: u32, arg3: u32) -> i32;
    fn blk_execute_rq(arg0: *mut request, arg1: bool) -> u8;
    fn blk_update_request(arg0: *mut request, arg1: u8, arg2: u32) -> bool;
    fn __blk_rq_map_sg(arg0: *mut request, arg1: *mut scatterlist, arg2: *mut *mut scatterlist) -> i32;
    #[link_name = "memcpy"]
    fn __builtin_memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn scsi_normalize_sense(arg0: *const u8, arg1: i32, arg2: *mut scsi_sense_hdr) -> bool;
    fn scsi_build_sense_buffer(arg0: i32, arg1: *mut u8, arg2: u8, arg3: u8, arg4: u8);
    fn sdev_prefix_printk(arg0: *const u8, arg1: *const scsi_device, arg2: *const u8, arg3: *const u8, ...);
    fn scmd_printk(arg0: *const u8, arg1: *mut scsi_cmnd, arg2: *const u8, ...);
    fn scsi_device_put(arg0: *mut scsi_device);
    fn starget_for_each_device(arg0: *mut scsi_target, arg1: *mut (), arg2: Option<unsafe extern "C" fn(*mut scsi_device, *mut ())>);
    fn __starget_for_each_device(arg0: *mut scsi_target, arg1: *mut (), arg2: Option<unsafe extern "C" fn(*mut scsi_device, *mut ())>);
    fn __scsi_iterate_devices(arg0: *mut Scsi_Host, arg1: *mut scsi_device) -> *mut scsi_device;
    fn scsi_is_target_device(arg0: *const device) -> i32;
    fn scsi_finish_command(arg0: *mut scsi_cmnd);
    fn scsi_print_command(arg0: *mut scsi_cmnd);
    fn scsi_print_sense(arg0: *mut scsi_cmnd);
    fn scsi_print_result(arg0: *mut scsi_cmnd, arg1: *const u8, arg2: i32);
    fn scsi_command_normalize_sense(arg0: *const scsi_cmnd, arg1: *mut scsi_sense_hdr) -> bool;
    fn __compiletime_assert_592();
    fn scsi_rescan_device(arg0: *mut scsi_device) -> i32;
    fn scsi_host_busy(arg0: *mut Scsi_Host) -> i32;
    fn __traceiter_scsi_dispatch_cmd_start(arg0: *mut (), arg1: *mut scsi_cmnd) -> i32;
    fn __SCT__tp_func_scsi_dispatch_cmd_start(arg0: *mut (), arg1: *mut scsi_cmnd) -> i32;
    fn __compiletime_assert_615();
    fn __traceiter_scsi_dispatch_cmd_error(arg0: *mut (), arg1: *mut scsi_cmnd, arg2: i32) -> i32;
    fn __SCT__tp_func_scsi_dispatch_cmd_error(arg0: *mut (), arg1: *mut scsi_cmnd, arg2: i32) -> i32;
    fn __compiletime_assert_622();
    fn __traceiter_scsi_dispatch_cmd_done(arg0: *mut (), arg1: *mut scsi_cmnd) -> i32;
    fn __SCT__tp_func_scsi_dispatch_cmd_done(arg0: *mut (), arg1: *mut scsi_cmnd) -> i32;
    fn __compiletime_assert_629();
    fn scsi_show_rq(arg0: *mut seq_file, arg1: *mut request);
    fn scmd_eh_abort_handler(arg0: *mut work_struct);
    fn scsi_timeout(arg0: *mut request) -> u32;
    fn scsi_decide_disposition(arg0: *mut scsi_cmnd) -> u32;
    fn scsi_eh_wakeup(arg0: *mut Scsi_Host, arg1: u32);
    fn scsi_eh_scmd_add(arg0: *mut scsi_cmnd);
    fn scsi_noretry_cmd(arg0: *mut scsi_cmnd) -> bool;
    fn scsi_eh_done(arg0: *mut scsi_cmnd);
    fn __compiletime_assert_658();
    fn __compiletime_assert_682();
    fn __compiletime_assert_720();
    fn __compiletime_assert_727();
    fn __compiletime_assert_731();
    fn __compiletime_assert_734();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.bswap.i16"]
    fn __slate_intrinsic_1a0c079c71348b00(arg0: u16) -> u16;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_a39e427d3b6c2df4(arg0: i32) -> bool;
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
struct io_bitmap {
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
union __SlateRecord_static_call_types_h_65 {
    r#type: u64,
    mods: *mut static_call_mod,
    sites: *mut static_call_site,
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

#[inline(always)]
fn __release_ctx_lock__1(var: *const raw_spinlock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__1(var: &raw_spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct raw_spinlock {
    raw_lock: qspinlock,
}


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
}


#[inline(always)]
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_key_mod {
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
fn arch_test_and_clear_bit(mut nr: i64, mut addr: *mut u64) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock  btrq  {2:r}, ({0:r})\nsetc {1}", in(reg) addr, lateout(reg_byte) __t0, in(reg) nr as u64, options(nostack, att_syntax));
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
fn instrument_write(v: *const (), size: u64) {
kasan_check_write(v, size as u32);
kcsan_check_write!(v, size);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct pollfd {
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
fn test_and_set_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_set_bit(nr, addr)
}

#[inline(always)]
fn test_and_clear_bit(nr: i64, addr: *mut u64) -> bool {
kcsan_mb!();
instrument_atomic_read_write(addr.wrapping_offset(BIT_WORD__2!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_and_clear_bit(nr, addr)
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[inline(always)]
fn ___clear_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___clear_bit(nr, addr);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
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
struct futex_private_hash {
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
    __slate_bits_0: __slate_bits::__SlateBits118U0,
}


#[inline(always)]
fn PTR_ERR(ptr: *const ()) -> i64 {
ptr as i64
}

#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_scsi_dispatch_cmd_start, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_scsi_dispatch_cmd_error, const 0 as u8, label {
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_scsi_dispatch_cmd_done, const 0 as u8, label {
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
    __slate_bits_0: __slate_bits::__SlateBits125U0,
}


#[inline(always)]
fn native_save_fl() -> u64 {
let mut flags: u64 = 0;
// "=rm" is safe here, because "pop" adjusts the stack before
// it evaluates its effective address -- this is part of the
// documented behavior of the "pop" instruction.
unsafe {
let __t0: u64;
core::arch::asm!("# __raw_save_flags\n\tpushf ; pop {0:r}", lateout(reg) __t0, options(nostack, att_syntax));
flags = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
// no input
flags
}

#[inline(always)]
fn arch_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
// Note for KASAN: we deliberately don't use READ_ONCE_NOCHECK() here,
// it's non-inlined function that increases binary size and stack usage.
__READ_ONCE!((*v).counter)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct seccomp_filter {
}


#[inline(always)]
fn arch_atomic_dec(mut v: *mut __SlateRecord_types_h_190) {
unsafe {
core::arch::asm!("lock decl ({0:r})", in(reg) core::ptr::addr_of_mut!((*v).counter), options(nostack, att_syntax));
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
fn arch_atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, mut __old: i32, mut __new: i32) -> i32 {
let mut __v0: i32;
let mut __ret: i32 = 0;
match (const { core::mem::size_of::<i32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u8;
unsafe {
let __t0: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u16;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t1 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u32;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t2 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = (unsafe { core::ptr::addr_of_mut!((*v).counter) }) as *mut u64;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, i32>(__t3 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v0 = __ret;
__v0
}

#[inline(always)]
fn arch_atomic64_add(mut i: i64, mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock addq {1:r}, ({0:r})\n# {2}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) i as u64, in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
}

#[inline(always)]
fn arch_atomic64_sub_and_test(mut i: i64, mut v: *mut __SlateRecord_types_h_197) -> bool {
let mut __v0: bool;
let mut c: bool = false;
unsafe {
let __t0: u8;
core::arch::asm!("lock subq {2:r}, ({0:r})\nsete {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), lateout(reg_byte) __t0, in(reg) i as u64, options(nostack, att_syntax));
c = (__t0 as u8) != 0;
}
__v0 = c;
__v0
}

#[inline(always)]
fn arch_atomic64_inc(mut v: *mut __SlateRecord_types_h_197) {
unsafe {
core::arch::asm!("lock incq ({0:r})\n# {1}", in(reg) core::ptr::addr_of_mut!((*v).counter), in(reg) core::ptr::addr_of!((*v).counter), options(nostack, att_syntax));
}
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
fn raw_atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
arch_atomic_read(v)
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
fn raw_atomic_dec(v: *mut __SlateRecord_types_h_190) {
arch_atomic_dec(v);
}

#[inline(always)]
fn raw_atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
raw_atomic_sub_return(1i32, v)
}

#[inline(always)]
fn raw_atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, old: i32, new: i32) -> i32 {
arch_atomic_cmpxchg(v, old, new)
}

#[inline(always)]
fn raw_atomic64_add(i: i64, v: *mut __SlateRecord_types_h_197) {
arch_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic64_inc(v: *mut __SlateRecord_types_h_197) {
arch_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic64_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
arch_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn raw_atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
raw_atomic64_add(i, v);
}

#[inline(always)]
fn raw_atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
raw_atomic64_inc(v);
}

#[inline(always)]
fn raw_atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
raw_atomic64_sub_and_test(i, v)
}

#[inline(always)]
fn atomic_read(v: *const __SlateRecord_types_h_190) -> i32 {
instrument_atomic_read(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_read(v)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn atomic_set(v: *mut __SlateRecord_types_h_190, i: i32) {
instrument_atomic_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_set(v, i);
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn atomic_inc(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc(v);
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
fn atomic_inc_return(v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_inc_return(v)
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
fn atomic_dec(v: *mut __SlateRecord_types_h_190) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec(v);
}

#[inline(always)]
fn atomic_dec_return(v: *mut __SlateRecord_types_h_190) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_dec_return(v)
}

#[inline(always)]
fn atomic_cmpxchg(v: *mut __SlateRecord_types_h_190, old: i32, new: i32) -> i32 {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_190>() as u64);
raw_atomic_cmpxchg(v, old, new)
}

#[inline(always)]
fn atomic_long_add(i: i64, v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_add(i, v);
}

#[inline(always)]
fn atomic_long_inc(v: *mut __SlateRecord_types_h_197) {
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_inc(v);
}

#[inline(always)]
fn atomic_long_sub_and_test(i: i64, v: *mut __SlateRecord_types_h_197) -> bool {
kcsan_mb!();
instrument_atomic_read_write(v as *const (), core::mem::size_of::<__SlateRecord_types_h_197>() as u64);
raw_atomic_long_sub_and_test(i, v)
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


// LIST_HEAD - definition of a &struct list_head with initialization values
// @name: name of the list_head
fn INIT_LIST_HEAD(list: *mut list_head) {
loop {
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_14() };
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
unsafe { __compiletime_assert_15() };
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

fn __list_del_entry_valid(entry: *mut list_head) -> bool {
true
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct audit_context {
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


fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
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


fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__2!() as isize) as *mut ()) as *mut list_head;
}
}

fn list_del_init(entry: *mut list_head) {
__list_del_entry(entry);
INIT_LIST_HEAD(entry);
}

fn list_move_tail(list: *mut list_head, head: *mut list_head) {
__list_del_entry(list);
list_add_tail(list, head);
}

fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
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


fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_20() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!((*head).next) as *const *mut list_head) };
(__v0 == (head as *mut list_head)) as i32
}

fn __list_splice(list: *const list_head, prev: *mut list_head, next: *mut list_head) {
let first: *mut list_head = unsafe { (*list).next };
let last: *mut list_head = unsafe { (*list).prev };
unsafe {
(*first).prev = prev;
}
unsafe {
(*prev).next = first;
}
unsafe {
(*last).next = next;
}
unsafe {
(*next).prev = last;
}
}

fn list_splice(list: *const list_head, head: *mut list_head) {
if list_empty(list) == 0 {
__list_splice(list, head, unsafe { (*head).next });
}
}

fn list_splice_init(list: *mut list_head, head: *mut list_head) {
if list_empty(list as *const list_head) == 0 {
__list_splice(list as *const list_head, head, unsafe { (*head).next });
INIT_LIST_HEAD(list);
}
}

// list_for_each	-	iterate over a list
// @pos:	the &struct list_head to use as a loop cursor.
// @head:	the head for your list.
// list_for_each_safe - iterate over a list safe against removal of list entry
// @pos:	the &struct list_head to use as a loop cursor.
// @n:		another &struct list_head to use as temporary storage
// @head:	the head for your list.
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

// for_each_cpu - iterate over every cpu in a mask
// @cpu: the (optionally unsigned) integer iterator
// @mask: the cpumask pointer
//
// After the loop, cpu is >= nr_cpu_ids.
#[inline(always)]
fn cpumask_test_cpu(cpu: i32, cpumask: *const cpumask) -> bool {
test_bit__2!(cpumask_check(cpu as u32), unsafe { (*cpumask).bits.as_ptr() as *const u64 }) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
}


#[inline(always)]
fn arch_local_save_flags() -> u64 {
native_save_fl()
}

#[inline(always)]
fn arch_irqs_disabled_flags(flags: u64) -> i32 {
(flags & _BITUL!(9i32) == 0) as i32
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
    __slate_bits_0: __slate_bits::__SlateBits126U0,
    pkru: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct vm_struct {
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
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_58: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

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
struct optimistic_spin_queue {
    tail: __SlateRecord_types_h_190,
}


#[inline(always)]
fn __acquire_ctx_lock__2(var: *const spinlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__2(var: &spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kmem_cache {
}


#[inline(always)]
fn __release_ctx_lock__2(var: &spinlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
fn __acquire_ctx_lock__3(var: &rwlock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct mem_cgroup {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct rwlock {
    raw_lock: qrwlock,
}


#[inline(always)]
fn __acquire_ctx_lock__4(var: &mutex) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__4(var: &mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[inline(always)]
fn __release_ctx_lock__4(var: &mutex) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uprobe {
}


#[inline(always)]
fn __release_shared_ctx_lock__4(var: &mutex) {
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
struct uprobe_xol_ops {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct llist_node {
    next: *mut llist_node,
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
struct plist_node {
    prio: i32,
    prio_list: list_head,
    node_list: list_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct xol_area {
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
struct ldt_struct {
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
struct seccomp {
    mode: i32,
    filter_count: __SlateRecord_types_h_190,
    filter: *mut seccomp_filter,
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
struct sigaction {
    sa_handler: Option<unsafe extern "C" fn(i32)>,
    sa_flags: u64,
    sa_restorer: Option<unsafe extern "C" fn()>,
    sa_mask: __SlateRecord_signal_h_25,
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
    __slate_bits_0: __slate_bits::__SlateBits223U0,
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
struct device_node {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[inline(always)]
fn spinlock_check(lock: *mut spinlock) -> *mut raw_spinlock {
unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) }
}

#[inline(always)]
fn spin_lock_irq(lock: *mut spinlock) {
raw_spin_lock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
}

#[inline(always)]
fn spin_unlock_irq(lock: *mut spinlock) {
raw_spin_unlock_irq!(unsafe { core::ptr::addr_of_mut!((*lock).__slate_anon_0.rlock) });
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
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
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
struct rseq_data {
    usrptr: *mut rseq,
    len: u32,
    sig: u32,
    event: rseq_event,
    ids: rseq_ids,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
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
struct pidfs_attr {
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct mm_cid_pcpu {
    cid: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_pid_h_62 {
    ino: u64,
    pidfs_hash: rhash_head,
    stashed: *mut dentry,
    attr: *mut pidfs_attr,
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


#[inline(always)]
fn __acquire_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__5(var: &seqlock) {
}

#[inline(always)]
fn __release_ctx_lock__5(var: &seqlock) {
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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
struct backing_dev_info {
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
let mut __UNIQUE_ID_guard_101: __SlateRecord_preempt_h_496 = class_preempt_constructor();
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_101) as *mut _;
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


fn __migrate_disable() {
let p: *mut task_struct = get_current();
if (unsafe { (*p).migration_disabled }) != 0 {
unsafe {
(*p).migration_disabled = ((*p).migration_disabled as i32).wrapping_add(1) as u16;
}
return;
}
let mut __UNIQUE_ID_guard_102: __SlateRecord_preempt_h_496 = class_preempt_constructor();
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_102) as *mut _;
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
fn migrate_enable() {
__migrate_enable();
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


fn mem_alloc_profiling_enabled() -> bool {
false
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
struct kref {
    refcount: refcount_struct,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct poll_table_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct swap_info_struct {
}


#[inline(always)]
fn __acquire_ctx_lock__6(var: &__ctx_lock_RCU) {
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


#[inline(always)]
fn __acquire_shared_ctx_lock__6(var: *const __ctx_lock_RCU) {
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


#[inline(always)]
fn __release_ctx_lock__6(var: &__ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct posix_acl {
}


#[inline(always)]
fn __release_shared_ctx_lock__6(var: *const __ctx_lock_RCU) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __ctx_lock_RCU {
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
struct rcu_gp_seq {
    norm: u64,
    exp: u64,
}


// Defined as a macro as it is a very low level header included from
// areas that don't even know about current.  This gives the rcu_read_lock()
// nesting depth, but makes sense only if CONFIG_PREEMPT_RCU -- in other
// types of kernel builds, the rcu_read_lock() nesting depth is unknowable.
fn init_rcu_head(head: *mut callback_head) {
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


fn destroy_rcu_head(head: *mut callback_head) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_uring_cmd {
}


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


fn rcu_read_unlock() {
RCU_LOCKDEP_WARN!(!unsafe { rcu_is_watching() });
rcu_lock_release!(); // Keep acq info for rls diags.
__release_shared!(unsafe { RCU });
unsafe { __rcu_read_unlock() };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_rcupdate_h_1216 {
    lock: *mut (),
}


#[inline(always)]
fn class_rcu_destructor(_T: *mut __SlateRecord_rcupdate_h_1216) {
rcu_read_unlock();
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


#[inline(always)]
fn class_rcu_constructor() -> __SlateRecord_rcupdate_h_1216 {
let _t: __SlateRecord_rcupdate_h_1216 = __SlateRecord_rcupdate_h_1216 { lock: 1i32 as *mut () };
rcu_read_lock();
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct kernfs_root {
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
fn __acquire_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__7(var: &rw_semaphore) {
}

#[inline(always)]
fn __release_ctx_lock__7(var: &rw_semaphore) {
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_kernfs_h_221 {
    dir: kernfs_elem_dir,
    symlink: kernfs_elem_symlink,
    attr: kernfs_elem_attr,
}


#[inline(always)]
fn __release_shared_ctx_lock__7(var: &rw_semaphore) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
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

fn schedule_work(work: *mut work_struct) -> bool {
queue_work(unsafe { system_percpu_wq }, work)
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

#[inline(always)]
fn __acquire_shared_ctx_lock__8(var: *const srcu_struct) {
}

#[inline(always)]
fn __release_ctx_lock__8(var: &srcu_struct) {
}

#[inline(always)]
fn __release_shared_ctx_lock__8(var: *const srcu_struct) {
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
struct wake_irq {
}


fn __srcu_read_lock_fast(ssp: *mut srcu_struct) -> *mut srcu_ctr {
let mut scp: *mut srcu_ctr = unsafe { core::mem::zeroed() };
let mut __v0: *mut srcu_ctr;
loop {
if !(__native_word__8!() || ((const { core::mem::size_of::<*mut srcu_ctr>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_168() };
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
__verify_pcpu_ptr__3!();
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
__verify_pcpu_ptr__2!();
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

fn __srcu_read_unlock_fast(ssp: *mut srcu_struct, mut scp: *mut srcu_ctr) {
__release_shared__2!(ssp);
unsafe {
core::arch::asm!("", options(nostack, att_syntax));
}
// Avoid leaking the critical section.
if IS_ENABLED!() == 0 {
loop {
__verify_pcpu_ptr__3!();
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
__verify_pcpu_ptr__2!();
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

#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_qos {
}


fn srcu_check_read_flavor(ssp: *mut srcu_struct, read_flavor: i32) {
if IS_ENABLED!() != 0 {
unsafe { __srcu_check_read_flavor(ssp, read_flavor) };
}
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
    __slate_bits_0: __slate_bits::__SlateBits428U0,
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
#[inline(always)]
fn __acquire_ctx_lock__9(var: &local_lock) {
}

#[inline(always)]
fn __acquire_shared_ctx_lock__9(var: &local_lock) {
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


#[inline(always)]
fn __release_ctx_lock__9(var: &local_lock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct disk_stats {
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
union __SlateRecord_blk_types_h_239 {
    bi_cookie: u32,
    __bi_nr_segments: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct blkcg_gq {
}


#[inline(always)]
fn __acquire_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bio_alloc_cache {
}


#[inline(always)]
fn __acquire_shared_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __release_ctx_lock__10(var: &local_trylock) {
}

#[inline(always)]
fn __release_shared_ctx_lock__10(var: &local_trylock) {
}

#[repr(C)]
#[derive(Clone, Copy)]
struct local_trylock {
    acquired: u8,
}


fn memalloc_flags_save(flags: u32) -> u32 {
let oldflags: u32 = !unsafe { (*get_current()).flags } & flags;
let __v0: *mut task_struct = get_current();
unsafe {
(*__v0).flags = (*__v0).flags | flags;
}
oldflags
}

fn memalloc_flags_restore(flags: u32) {
let __v0: *mut task_struct = get_current();
let __v2: u32 = (unsafe { (*__v0).flags }) & !flags;
unsafe {
(*__v0).flags = __v2;
}
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


fn memalloc_noio_save() -> u32 {
memalloc_flags_save(524288u32)
}

fn memalloc_noio_restore(flags: u32) {
memalloc_flags_restore(flags);
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
struct blk_mq_ctx {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct list_lru {
    node: *mut list_lru_node,
}


#[inline(always)]
fn __acquire_ctx_lock__11(var: &__context_bitlock) {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct __context_bitlock {
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


#[inline(always)]
fn pagefault_disabled_inc() {
let __v0: *mut task_struct = get_current();
let __v1: i32 = unsafe { (*__v0).pagefault_disabled };
unsafe {
(*__v0).pagefault_disabled = __v1.wrapping_add(1);
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct bsg_device {
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
    pub struct __SlateBits576U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits678U0 {
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
    pub struct __SlateBits675U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits678U1 {
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
    pub struct __SlateBits678U2 {
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
    pub struct __SlateBits678U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits649U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits590U0 {
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
    pub struct __SlateBits125U0 {
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
    pub struct __SlateBits428U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    pub struct __SlateBits118U0 {
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
    pub struct __SlateBits126U0 {
        #[bits(1)]
        pub iopl_warn: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits223U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits836U0 {
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
    pub struct __SlateBits848U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits846U0 {
        #[bits(1)]
        pub create_work_queue: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
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
    #[bitfields::bitfield([u8; 8], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits834U0 {
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
    pub struct __SlateBits829U0 {
        #[bits(1)]
        pub longlba: u8,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
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
    __slate_bits_0: __slate_bits::__SlateBits576U0,
    rcu: callback_head,
    r#ref: *mut percpu_ref,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref {
    percpu_count_ptr: u64,
    data: *mut percpu_ref_data,
}


fn __ref_is_percpu(r#ref: *mut percpu_ref, percpu_countp: &mut *mut u64) -> bool {
let mut percpu_ptr: u64 = 0;
// The value of @ref->percpu_count_ptr is tested for
// !__PERCPU_REF_ATOMIC, which may be set asynchronously, and then
// used as a pointer.  If the compiler generates a separate fetch
// when using it as a pointer, __PERCPU_REF_ATOMIC may be set in
// between contaminating the pointer value, meaning that
// READ_ONCE() is required when fetching it.
//
// The dependency ordering from the READ_ONCE() pairs
// with smp_store_release() in __percpu_ref_switch_to_percpu().
let mut __v0: u64;
loop {
if !(__native_word__3!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_386() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*r#ref).percpu_count_ptr) as *const u64) };
percpu_ptr = __v0;
// Theoretically, the following could test just ATOMIC; however,
// then we'd have to mask off DEAD separately as DEAD may be
// visible without ATOMIC if we race with percpu_ref_kill().  DEAD
// implies ATOMIC anyway.  Test them together.
if unlikely__4!(percpu_ptr & 3) != 0 {
return false;
}
unsafe {
*(percpu_countp as *mut *mut u64) = percpu_ptr as *mut u64;
}
true
}

fn percpu_ref_get_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr & 255) as u8;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr & 65535) as u16;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr) }) && (nr == 1 || nr == (1i32.wrapping_neg() as u64)) { nr as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr;
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr;
if false {
let pto_tmp__: u64 = nr;
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
atomic_long_add(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) });
}
rcu_read_unlock();
}

fn percpu_ref_get(r#ref: *mut percpu_ref) {
percpu_ref_get_many(r#ref, 1u64);
}

fn percpu_ref_put_many(r#ref: *mut percpu_ref, nr: u64) {
let mut percpu_count: *mut u64 = unsafe { core::mem::zeroed() };
rcu_read_lock();
if __ref_is_percpu(r#ref, unsafe { &mut (*core::ptr::addr_of_mut!(percpu_count)) }) {
loop {
__verify_pcpu_ptr__5!();
match (const { core::mem::size_of::<u64>() }) as u64 {
1 => {
loop {
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decb %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u8 = (nr.wrapping_neg() & 255) as u8;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addb {1:l}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) (pto_val__ as u8) as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decw %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u16 = (nr.wrapping_neg() & 65535) as u16;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addw {1:x}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u16, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decl %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u32 = (nr.wrapping_neg() & 4294967295) as u32;
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addl {1:e}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u32, options(nostack, att_syntax));
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
let pao_ID__: i32 = if (unsafe { __slate_intrinsic_f217c4720f04f6db(nr.wrapping_neg()) }) && (nr.wrapping_neg() == 1 || nr.wrapping_neg() == (1i32.wrapping_neg() as u64)) { nr.wrapping_neg() as i32 } else { 0 };
if false {
let pao_tmp__: u64 = nr.wrapping_neg();
}
if pao_ID__ == 1 {
unsafe {
core::arch::asm!("incq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
if pao_ID__ == 1i32.wrapping_neg() {
unsafe {
core::arch::asm!("decq %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, options(nostack, att_syntax));
}
} else {
loop {
let mut pto_val__: u64 = nr.wrapping_neg();
if false {
let pto_tmp__: u64 = nr.wrapping_neg();
}
unsafe {
core::arch::asm!("addq {1:r}, %gs:({0:r})", in(reg) (percpu_count as u64) as *mut u64, in(reg) pto_val__ as u64, options(nostack, att_syntax));
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
if unlikely!(atomic_long_sub_and_test(nr as i64, unsafe { core::ptr::addr_of_mut!((*(*r#ref).data).count) })) != 0 {
unsafe { unsafe { (*(*r#ref).data).release }.unwrap()(r#ref) };
}
}
rcu_read_unlock();
}

fn percpu_ref_put(r#ref: *mut percpu_ref) {
percpu_ref_put_many(r#ref, 1u64);
}

// define SLAB_HWCACHE_ALIGN - Align objects on cache line boundaries.
//
// Sufficiently large objects are aligned on cache line boundary. For object
// size smaller than a half of cache line size, the alignment is on the half of
// cache line size. In general, if object size is smaller than 1/2^n of cache
// line size, the alignment is adjusted to 1/2^n.
//
// If explicit alignment is also requested by the respective
// &struct kmem_cache_args field, the greater of both is alignments is applied.
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


fn kmem_cache_create_usercopy(name: *const u8, size: u32, align: u32, flags: u32, useroffset: u32, usersize: u32, ctor: Option<unsafe extern "C" fn(*mut ())>) -> *mut kmem_cache {
let mut kmem_args: kmem_cache_args = {
let mut __t0: kmem_cache_args = unsafe { core::mem::zeroed() };
__t0.align = align;
__t0.useroffset = useroffset;
__t0.usersize = usersize;
__t0.ctor = ctor;
__t0
};
unsafe { __kmem_cache_create_args(name, size, core::ptr::addr_of_mut!(kmem_args), flags) }
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

core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_scsi_failures_reset_retries: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_execute_cmd: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {1} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_free_sgtables: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {2} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_alloc_sgtables: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {3} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_alloc_request: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {4} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_done: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {5} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_done_direct: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {6} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_init_limits: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {7} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_get_internal_cmd: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {8} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_put_internal_cmd: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {9} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_block_requests: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {10} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_unblock_requests: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {11} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_mode_select: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {12} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_mode_sense: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {13} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_test_unit_ready: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {14} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_device_set_state: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {15} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_sdev_evt_send: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {16} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_sdev_evt_alloc: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {17} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_sdev_evt_send_simple: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {18} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_device_quiesce: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {19} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_device_resume: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {20} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_target_quiesce: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {21} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_target_resume: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {22} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_internal_device_block_nowait: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {23} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_internal_device_unblock_nowait: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {24} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_block_targets: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {25} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_target_unblock: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {26} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_host_block: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {27} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_host_unblock: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {28} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_kmap_atomic_sg: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {29} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_kunmap_atomic_sg: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {30} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_sdev_disable_disk_events: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {31} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_sdev_enable_disk_events: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {32} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_vpd_lun_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {33} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_vpd_lun_serial: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {34} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_vpd_tpg_id: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {35} ; .previous\n.section \".export_symbol\",\"a\" ; __export_symbol_scsi_build_sense: ; .asciz \"GPL\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {36} ; .previous", sym scsi_failures_reset_retries, sym scsi_execute_cmd, sym scsi_free_sgtables, sym scsi_alloc_sgtables, sym scsi_alloc_request, sym scsi_done, sym scsi_done_direct, sym scsi_init_limits, sym scsi_get_internal_cmd, sym scsi_put_internal_cmd, sym scsi_block_requests, sym scsi_unblock_requests, sym scsi_mode_select, sym scsi_mode_sense, sym scsi_test_unit_ready, sym scsi_device_set_state, sym sdev_evt_send, sym sdev_evt_alloc, sym sdev_evt_send_simple, sym scsi_device_quiesce, sym scsi_device_resume, sym scsi_target_quiesce, sym scsi_target_resume, sym scsi_internal_device_block_nowait, sym scsi_internal_device_unblock_nowait, sym scsi_block_targets, sym scsi_target_unblock, sym scsi_host_block, sym scsi_host_unblock, sym scsi_kmap_atomic_sg, sym scsi_kunmap_atomic_sg, sym sdev_disable_disk_events, sym sdev_enable_disk_events, sym scsi_vpd_lun_id, sym scsi_vpd_lun_serial, sym scsi_vpd_tpg_id, sym scsi_build_sense, options(att_syntax));

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


fn page_range_contiguous(page: &page, nr_pages: u64) -> bool {
true
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
    __slate_bits_0: __slate_bits::__SlateBits649U0,
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
    __slate_bits_0: __slate_bits::__SlateBits590U0,
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
static mut __UNIQUE_ID_addressable___SCK__preempt_schedule_486: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__preempt_schedule) }) as *mut ();

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
struct pm_subsys_data {
    lock: spinlock,
    refcount: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct dev_pm_info {
    power_state: pm_message,
    __slate_bits_0: __slate_bits::__SlateBits678U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits678U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits678U2,
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
    __slate_bits_3: __slate_bits::__SlateBits678U3,
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
    __slate_bits_0: __slate_bits::__SlateBits675U0,
}


fn dev_name(dev: *const device) -> *const u8 {
// Use the init name until the kobject becomes available
if !unsafe { (*dev).init_name }.is_null() {
return unsafe { (*dev).init_name };
}
kobject_name(unsafe { core::ptr::addr_of!((*dev).kobj) })
}

fn dev_to_node(dev: *mut device) -> i32 {
unsafe { (*dev).numa_node }
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
// BLK_STS_DEV_RESOURCE is returned from the driver to the block layer if
// device related resources are unavailable, but the driver can guarantee
// that the queue will be rerun in the future once resources become
// available again. This is typically the case for device specific
// resources that are consumed for IO. If the driver fails allocating these
// resources, we know that inflight (or pending) IO will free these
// resource upon completion.
//
// This is different from BLK_STS_RESOURCE in that it explicitly references
// a device specific resource. For resources of wider scope, allocation
// failure can happen without having pending IO. This means that we can't
// rely on request completions freeing these resources, as IO may not be in
// flight. Examples of that are kernel memory allocations, DMA mappings, or
// any other system wide resources.
// BLK_STS_ZONE_OPEN_RESOURCE is returned from the driver in the completion
// path if the device returns a status indicating that too many zone resources
// are currently open. The same command should be successful if resubmitted
// after the number of open zones decreases below the device's limits, which is
// reported in the request_queue's max_open_zones.
// BLK_STS_OFFLINE is returned from the driver when the target device is offline
// or is being taken offline. This could help differentiate the case where a
// device is intentionally being shut down from a real I/O error.
// BLK_STS_DURATION_LIMIT is returned from the driver when the target device
// aborted the command because it exceeded one of its Command Duration Limits.
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


fn sbitmap_deferred_clear_bit(sb: *mut sbitmap, bitnr: u32) {
let addr: *mut u64 = unsafe { core::ptr::addr_of_mut!((*unsafe { (*sb).map }.wrapping_offset(SB_NR_TO_INDEX!(sb, bitnr) as isize)).cleared) };
set_bit(SB_NR_TO_BIT!(sb, bitnr) as i64, addr as *mut u64);
}

fn sbitmap_put(sb: *mut sbitmap, bitnr: u32) {
sbitmap_deferred_clear_bit(sb, bitnr);
if likely!(!unsafe { (*sb).alloc_hint }.is_null() && !unsafe { (*sb).round_robin } && bitnr < unsafe { (*sb).depth }) != 0 {
let mut __v0: *mut u32;
__verify_pcpu_ptr__6!();
let mut __v2: *mut u32;
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
tcp_ptr__ = tcp_ptr__.wrapping_add((unsafe { (*sb).alloc_hint }) as u64);
__v2 = tcp_ptr__ as *mut u32;
__v0 = __v2;
unsafe {
*__v0 = bitnr;
}
}
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
// contributes to the random number pool
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

fn sg_assign_page(sg: *mut scatterlist, page: *mut page) {
let page_link: u64 = (unsafe { (*sg).page_link }) & (1 | 2);
// In order for the low bit stealing approach to work, pages
// must be aligned at a 32-bit boundary as a minimum.
loop {
if unlikely__4!((page as u64) & (1 | 2)) != 0 {
loop {
unsafe {
core::arch::asm!("570: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 570b - ., 3; .popsection\n# {0}", const 570 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_sg_assign_page_0, sym __slate_asm_string_sg_assign_page_1, const 137 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe {
(*sg).page_link = page_link | (page as u64);
}
}

fn sg_set_page(sg: *mut scatterlist, page: *mut page, len: u32, offset: u32) {
VM_WARN_ON_ONCE!();
sg_assign_page(sg, page);
unsafe {
(*sg).offset = offset;
}
unsafe {
(*sg).length = len;
}
}

fn sg_page(sg: *mut scatterlist) -> *mut page {
((unsafe { (*sg).page_link }) & !(1 | 2)) as *mut page
}

fn sg_set_buf(sg: *mut scatterlist, buf: *const (), buflen: u32) {
sg_set_page(sg, virt_to_page!(buf), buflen, offset_in_page!(buf) as u32);
}

// Loop over each sg element, following the pointer to a new list if necessary
fn sg_mark_end(sg: *mut scatterlist) {
// Set termination bit, clear potential chain bit
unsafe {
(*sg).page_link = (*sg).page_link | 2;
}
let __v5: u64 = (unsafe { (*sg).page_link }) & !1;
unsafe {
(*sg).page_link = __v5;
}
}

fn sg_unmark_end(__v0: *mut scatterlist) {
let __v2: u64 = (unsafe { (*__v0).page_link }) & !2;
unsafe {
(*__v0).page_link = __v2;
}
}

// DMA_ATTR_NO_WARN: This tells the DMA-mapping subsystem to suppress
// allocation failure reports (similarly to __GFP_NOWARN).
// DMA_ATTR_MMIO - Indicates memory-mapped I/O (MMIO) region for DMA mapping
//
// This attribute indicates the physical address is not normal system
// memory. It may not be used with kmap*()/phys_to_virt()/phys_to_page()
// functions, it may not be cacheable, and access using CPU load/store
// instructions may not be allowed.
//
// Usually this will be used to describe MMIO addresses, or other non-cacheable
// register addresses. When DMA mapping this sort of address we call
// the operation Peer to Peer as a one device is DMA'ing to another device.
// For PCI devices the p2pdma APIs must be used to determine if DMA_ATTR_MMIO
// is appropriate.
//
// For architectures that require cache flushing for DMA coherence
// DMA_ATTR_MMIO will not perform any cache flushing. The address
// provided must never be mapped cacheable into the CPU.
// A dma_addr_t can hold any valid DMA or bus address for the platform.  It can
// be given to a device to use as a DMA source or target.  It is specific to a
// given device and there may be a translation between the CPU physical address
// space and the bus address space.
//
// DMA_MAPPING_ERROR is the magic error code if a mapping failed.  It should not
// be used directly in drivers, but checked for using dma_mapping_error()
// instead.
// Use the high bit to mark if we used swiotlb for one or more ranges.
fn dma_set_max_seg_size(dev: *mut device, size: u32) {
let mut __v0: i64;
let __ret_warn_on: i32 = unsafe { (*dev).dma_parms }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("577: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 577b - ., 3; .popsection\n# {0}", const 577 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dma_set_max_seg_size_1, sym __slate_asm_string_dma_set_max_seg_size_2, const 676 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("578: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 578b - ., 4; .popsection\n# {0}", const 578 as i32, options(nostack, att_syntax));
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
unsafe {
(*(*dev).dma_parms).max_segment_size = size;
}
}

fn dma_set_seg_boundary(dev: *mut device, mask: u64) {
let mut __v0: i64;
let __ret_warn_on: i32 = unsafe { (*dev).dma_parms }.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("579: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 579b - ., 3; .popsection\n# {0}", const 579 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_dma_set_seg_boundary_1, sym __slate_asm_string_dma_set_seg_boundary_2, const 709 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("580: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 580b - ., 4; .popsection\n# {0}", const 580 as i32, options(nostack, att_syntax));
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
unsafe {
(*(*dev).dma_parms).segment_boundary_mask = mask;
}
}

fn dma_get_cache_alignment() -> i32 {
1
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


fn req_op(req: *const request) -> u32 {
((unsafe { (*req).cmd_flags }) & 255u32) as u32
}

fn blk_rq_is_passthrough(rq: *mut request) -> bool {
blk_op_is_passthrough(unsafe { (*rq).cmd_flags })
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


fn blk_mq_complete_request_direct(rq: *mut request, complete: Option<unsafe extern "C" fn(*mut request)>) {
loop {
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_585() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__WRITE_ONCE__7!((*rq).state, 2i32);
if true {
break;
}
}
unsafe { complete.unwrap()(rq) };
}

fn blk_mq_is_reserved_rq(rq: *mut request) -> bool {
(unsafe { (*rq).rq_flags }) & 32768u32 != 0
}

fn blk_mq_freeze_queue(q: *mut request_queue) -> u32 {
let memflags: u32 = memalloc_noio_save();
unsafe { blk_mq_freeze_queue_nomemsave(q) };
memflags
}

fn blk_mq_unfreeze_queue(q: *mut request_queue, memflags: u32) {
unsafe { blk_mq_unfreeze_queue_nomemrestore(q) };
memalloc_noio_restore(memflags);
}

fn blk_should_fake_timeout(q: *mut request_queue) -> bool {
let __v0: bool = IS_ENABLED!() != 0 && test_bit!(3i32, unsafe { core::ptr::addr_of_mut!((*q).queue_flags) }) != 0;
if __v0 {
return unsafe { __blk_should_fake_timeout(q) };
}
false
}

fn blk_mq_rq_from_pdu(pdu: *mut ()) -> *mut request {
((unsafe { (pdu as *const u8).wrapping_offset(-(core::mem::size_of::<request>() as isize)) }) as *mut ()) as *mut request
}

fn blk_mq_rq_to_pdu(rq: *mut request) -> *mut () {
rq.wrapping_offset(1 as isize) as *mut ()
}

fn blk_rq_bytes(rq: *const request) -> u32 {
unsafe { (*rq).__data_len }
}

fn blk_rq_payload_bytes(rq: *mut request) -> u32 {
if (unsafe { (*rq).rq_flags }) & 4096u32 != 0 {
return unsafe { (*rq).__slate_anon_25.special_vec.bv_len };
}
blk_rq_bytes(rq as *const request)
}

fn blk_rq_nr_phys_segments(rq: *mut request) -> u16 {
if (unsafe { (*rq).rq_flags }) & 4096u32 != 0 {
return 1;
}
unsafe { (*rq).nr_phys_segments }
}

fn blk_rq_map_integrity_sg(q: *mut request, s: *mut scatterlist) -> i32 {
0
}

fn blk_integrity_rq(rq: *const request) -> bool {
false
}

fn get_unaligned_be16(p: *const ()) -> u16 {
__swab16!(__get_unaligned_t__2!(u16, p))
}

fn put_unaligned_be16(val: u16, p: *mut ()) {
__put_unaligned_t!(u16, __swab16!(val), p);
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

// The Well Known LUNS (SAM-3) in our int representation of a LUN
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
    __slate_bits_0: __slate_bits::__SlateBits829U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_event {
    evt_type: u32,
    node: list_head,
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
    __slate_bits_0: __slate_bits::__SlateBits834U0,
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
    __slate_bits_0: __slate_bits::__SlateBits836U0,
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

// shost_for_each_device - iterate over all devices of a host
// @sdev: the &struct scsi_device to use as a cursor
// @shost: the &struct scsi_host to iterate over
//
// Iterator that returns each device attached to @shost.  This loop
// takes a reference on each device and releases it at the end.  If
// you break out of the loop, you must call scsi_device_put(sdev).
// __shost_for_each_device - iterate over all devices of a host (UNLOCKED)
// @sdev: the &struct scsi_device to use as a cursor
// @shost: the &struct scsi_host to iterate over
//
// Iterator that returns each device attached to @shost.  It does _not_
// take a reference on the scsi_device, so the whole loop must be
// protected by shost->host_lock.
//
// Note: The only reason to use this is because you need to access the
// device list in interrupt context.  Otherwise you really want to use
// shost_for_each_device instead.
// scsi_execute_cmd users can set scsi_failure.result to have
// scsi_check_passthrough fail/retry a command. scsi_failure.result can be a
// specific host byte or message code, or SCMD_FAILURE_RESULT_ANY can be used
// to match any host or message code.
// Set scsi_failure.result to SCMD_FAILURE_STAT_ANY to fail/retry any failure
// scsi_status_is_good returns false for.
// The following can be set to the scsi_failure sense, asc and ascq fields to
// match on any sense, ASC, or ASCQ value.
// Always retry a matching failure.
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


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_vpd_lun_id")]
extern "C" fn scsi_vpd_lun_id(sdev: *mut scsi_device, id: *mut u8, id_len: u64) -> i32 {
let mut cur_id_prio: u8 = 0;
let mut cur_id_size: u8 = 0;
let mut d: *const u8 = unsafe { core::mem::zeroed() };
let mut cur_id_str: *const u8 = unsafe { core::mem::zeroed() };
let mut vpd_pg83: *const scsi_vpd = unsafe { core::mem::zeroed() };
let mut id_size: i32 = 22i32.wrapping_neg();
rcu_read_lock();
let mut __v0: *mut scsi_vpd;
let mut __v1: *mut scsi_vpd;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_727() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sdev).vpd_pg83) as *const *mut scsi_vpd) };
let __UNIQUE_ID_rcu_726: *mut scsi_vpd = __v1;
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
__v0 = __UNIQUE_ID_rcu_726;
vpd_pg83 = __v0 as *const scsi_vpd;
if vpd_pg83.is_null() {
rcu_read_unlock();
return 6i32.wrapping_neg();
}
// The id string must be at least 20 bytes + terminating NULL byte
if id_len < 21 {
rcu_read_unlock();
return 22i32.wrapping_neg();
}
unsafe { memset(id as *mut (), 0i32, id_len) };
d = unsafe { core::ptr::addr_of!((*vpd_pg83).data) as *const u8 }.wrapping_offset(4 as isize);
while d < unsafe { core::ptr::addr_of!((*vpd_pg83).data) as *const u8 }.wrapping_offset((unsafe { (*vpd_pg83).len }) as isize) {
let mut prio: u8 = designator_prio(d);
if !((prio as i32) == 0 || (cur_id_prio as i32) > (prio as i32)) {
'__slate_break_7: {
match ((unsafe { *d.wrapping_offset(1 as isize) }) as i32) & 15 {
1 => {
if (cur_id_size as i32) <= ((unsafe { *d.wrapping_offset(3 as isize) }) as i32) {
// T10 Vendor ID
cur_id_prio = prio;
cur_id_size = unsafe { *d.wrapping_offset(3 as isize) };
if ((cur_id_size as i32).wrapping_add(4) as u64) > id_len {
cur_id_size = id_len.wrapping_sub(4) as u8;
}
cur_id_str = d.wrapping_offset(4 as isize);
id_size = unsafe { snprintf(id, id_len, c"t10.%*pE".as_ptr() as *const u8, cur_id_size as i32, cur_id_str) };
}
}
2 => {
cur_id_prio = prio; // EUI-64
cur_id_size = unsafe { *d.wrapping_offset(3 as isize) };
cur_id_str = d.wrapping_offset(4 as isize);
match cur_id_size as i32 {
8 => {
id_size = unsafe { snprintf(id, id_len, c"eui.%8phN".as_ptr() as *const u8, cur_id_str) };
}
12 => {
id_size = unsafe { snprintf(id, id_len, c"eui.%12phN".as_ptr() as *const u8, cur_id_str) };
}
16 => {
id_size = unsafe { snprintf(id, id_len, c"eui.%16phN".as_ptr() as *const u8, cur_id_str) };
}
_ => {
}
}
}
3 => {
cur_id_prio = prio; // NAA
cur_id_size = unsafe { *d.wrapping_offset(3 as isize) };
cur_id_str = d.wrapping_offset(4 as isize);
match cur_id_size as i32 {
8 => {
id_size = unsafe { snprintf(id, id_len, c"naa.%8phN".as_ptr() as *const u8, cur_id_str) };
}
16 => {
id_size = unsafe { snprintf(id, id_len, c"naa.%16phN".as_ptr() as *const u8, cur_id_str) };
}
_ => {
}
}
}
8 => {
if (cur_id_size as i32) <= ((unsafe { *d.wrapping_offset(3 as isize) }) as i32) {
// SCSI name string
// Prefer others for truncated descriptor
if (((unsafe { *d.wrapping_offset(3 as isize) }) as i32) as u64) > id_len {
prio = 2;
if (cur_id_prio as i32) > (prio as i32) {
break '__slate_break_7;
}
}
cur_id_prio = prio;
let __v10: i32 = (unsafe { *d.wrapping_offset(3 as isize) }) as i32;
id_size = __v10;
cur_id_size = __v10 as u8;
cur_id_str = d.wrapping_offset(4 as isize);
if ((cur_id_size as i32) as u64) >= id_len {
cur_id_size = id_len.wrapping_sub(1) as u8;
}
unsafe { memcpy(id as *mut (), cur_id_str as *const (), cur_id_size as u64) };
}
}
_ => {
}
}
}
}
let __v11: *const u8 = d;
d = __v11.wrapping_offset(((unsafe { *d.wrapping_offset(3 as isize) }) as i32).wrapping_add(4) as isize);
}
rcu_read_unlock();
id_size
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_vpd_lun_serial")]
extern "C" fn scsi_vpd_lun_serial(sdev: *mut scsi_device, sn: *mut u8, sn_size: u64) -> i32 {
let mut vpd_pg80: *const scsi_vpd = unsafe { core::mem::zeroed() };
let mut d: *const u8 = unsafe { core::mem::zeroed() };
let mut len: i32 = 0;
let mut __UNIQUE_ID_guard_729: __SlateRecord_rcupdate_h_1216 = class_rcu_constructor();
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_729) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_rcu_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __v1: *mut scsi_vpd;
let mut __v2: *mut scsi_vpd;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_731() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v2 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sdev).vpd_pg80) as *const *mut scsi_vpd) };
let __UNIQUE_ID_rcu_730: *mut scsi_vpd = __v2;
loop {
let __v5: bool = if false {
let __v6: bool = false || rcu_read_lock_held() != 0;
!__v6
} else {
false
};
if !__v5 {
break;
}
}
__v1 = __UNIQUE_ID_rcu_730;
vpd_pg80 = __v1 as *const scsi_vpd;
if vpd_pg80.is_null() {
return 6i32.wrapping_neg();
}
len = unsafe { (*vpd_pg80).len }.wrapping_sub(4);
d = unsafe { core::ptr::addr_of!((*vpd_pg80).data) as *const u8 }.wrapping_offset(4 as isize);
// Skip leading spaces
while len > 0 && isspace!(unsafe { *d }) {
len = len.wrapping_sub(1);
let __v9: *const u8 = d;
d = __v9.wrapping_offset(1 as isize);
}
// Skip trailing spaces
while len > 0 && isspace!(unsafe { *d.wrapping_offset(len.wrapping_sub(1) as isize) }) {
len = len.wrapping_sub(1);
}
if sn_size < (len.wrapping_add(1) as u64) {
return 22i32.wrapping_neg();
}
unsafe { memcpy(sn as *mut (), d as *const (), len as u64) };
unsafe {
*sn.wrapping_offset(len as isize) = 0;
}
len
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_vpd_tpg_id")]
extern "C" fn scsi_vpd_tpg_id(sdev: *mut scsi_device, rel_id: *mut i32) -> i32 {
let mut d: *const u8 = unsafe { core::mem::zeroed() };
let mut vpd_pg83: *const scsi_vpd = unsafe { core::mem::zeroed() };
let mut group_id: i32 = 11i32.wrapping_neg();
let mut rel_port: i32 = 1i32.wrapping_neg();
rcu_read_lock();
let mut __v0: *mut scsi_vpd;
let mut __v1: *mut scsi_vpd;
loop {
if !(__native_word!() || ((const { core::mem::size_of::<*mut scsi_vpd>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_734() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*sdev).vpd_pg83) as *const *mut scsi_vpd) };
let __UNIQUE_ID_rcu_733: *mut scsi_vpd = __v1;
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
__v0 = __UNIQUE_ID_rcu_733;
vpd_pg83 = __v0 as *const scsi_vpd;
if vpd_pg83.is_null() {
rcu_read_unlock();
return 6i32.wrapping_neg();
}
d = unsafe { core::ptr::addr_of!((*vpd_pg83).data) as *const u8 }.wrapping_offset(4 as isize);
while d < unsafe { core::ptr::addr_of!((*vpd_pg83).data) as *const u8 }.wrapping_offset((unsafe { (*vpd_pg83).len }) as isize) {
match ((unsafe { *d.wrapping_offset(1 as isize) }) as i32) & 15 {
4 => {
rel_port = get_unaligned_be16(d.wrapping_offset(6 as isize) as *const ()) as i32; // Relative target port
}
5 => {
group_id = get_unaligned_be16(d.wrapping_offset(6 as isize) as *const ()) as i32; // Target port group
}
_ => {
}
}
let __v7: *const u8 = d;
d = __v7.wrapping_offset(((unsafe { *d.wrapping_offset(3 as isize) }) as i32).wrapping_add(4) as isize);
}
rcu_read_unlock();
if group_id >= 0 && !rel_id.is_null() && rel_port != 1i32.wrapping_neg() {
unsafe {
*rel_id = rel_port;
}
}
group_id
}

fn scsi_device_online(sdev: *mut scsi_device) -> i32 {
(((unsafe { (*sdev).sdev_state }) as u32) != 6 && ((unsafe { (*sdev).sdev_state }) as u32) != 7 && ((unsafe { (*sdev).sdev_state }) as u32) != 4) as i32
}

fn scsi_device_blocked(sdev: *mut scsi_device) -> i32 {
(((unsafe { (*sdev).sdev_state }) as u32) == 8 || ((unsafe { (*sdev).sdev_state }) as u32) == 9) as i32
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


// for scmd->flags
// flags preserved across unprep / reprep
// for scmd->state
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

fn scsi_cmd_priv(cmd: *mut scsi_cmnd) -> *mut () {
cmd.wrapping_offset(1 as isize) as *mut ()
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_kunmap_atomic_sg")]
extern "C" fn scsi_kunmap_atomic_sg(virt: *mut ()) {
loop {
loop {
if __same_type!() != 0 {
unsafe { __compiletime_assert_720() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__kunmap_atomic(virt as *const ());
if true {
break;
}
}
}

fn scsi_set_resid(cmd: *mut scsi_cmnd, resid: u32) {
unsafe {
(*cmd).resid_len = resid;
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


fn scsi_cmd_to_driver(cmd: *mut scsi_cmnd) -> *mut scsi_driver {
to_scsi_driver!(unsafe { (*(*cmd).device).sdev_gendev.driver })
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
    __slate_bits_0: __slate_bits::__SlateBits848U0,
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


fn scsi_get_host_state(shost: *mut Scsi_Host) -> u32 {
let mut __v0: u32;
disable_context_analysis!();
let mut __v1: u32;
loop {
if !(__native_word__6!() || ((const { core::mem::size_of::<u32>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_592() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v1 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*shost).shost_state) as *const u32) };
__v0 = __v1;
__v0
}

fn scsi_host_in_recovery(shost: *mut Scsi_Host) -> i32 {
let state: u32 = scsi_get_host_state(shost);
((state as u32) == 5 || (state as u32) == 6 || (state as u32) == 7 || ((unsafe { (*shost).__slate_bits_0.__get_tmf_in_progress() }) as i32) != 0) as i32
}

fn scsi_host_get_prot(shost: *mut Scsi_Host) -> u32 {
unsafe { (*shost).prot_capabilities }
}

fn scsi_host_prot_dma(shost: *mut Scsi_Host) -> i32 {
((unsafe { (*shost).prot_capabilities }) >= 8) as i32
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
    __slate_bits_0: __slate_bits::__SlateBits846U0,
    eh_strategy_handler: Option<unsafe extern "C" fn(*mut Scsi_Host)>,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_init_limits")]
extern "C" fn scsi_init_limits(shost: *mut Scsi_Host, lim: *mut queue_limits) {
let dev: *mut device = unsafe { (*shost).dma_dev };
unsafe { memset(lim as *mut (), 0i32, core::mem::size_of::<queue_limits>() as u64) };
unsafe {
(*lim).max_segments = min_t__4!(u16, unsafe { (*shost).sg_tablesize }, 2048i32) as u16;
}
if scsi_host_prot_dma(shost) != 0 {
let mut __v3: i32;
let __x: u16 = unsafe { (*shost).sg_prot_tablesize };
let __y: u16 = 65535;
let __v4: i32 = if (__x as i32) == 0 {
__y as i32
} else {
let __v5: i32 = if (__y as i32) == 0 {
__x as i32
} else {
let mut __v6: i32;
let __UNIQUE_ID_x__680: u16 = __x;
let __UNIQUE_ID_y__681: u16 = __y;
loop {
if __types_ok__4!(__UNIQUE_ID_x__680, __UNIQUE_ID_y__681) == 0 {
unsafe { __compiletime_assert_682() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v6 = if (__UNIQUE_ID_x__680 as i32) < (__UNIQUE_ID_y__681 as i32) { __UNIQUE_ID_x__680 as i32 } else { __UNIQUE_ID_y__681 as i32 };
__v6
};
__v5
};
__v3 = __v4;
unsafe {
(*shost).sg_prot_tablesize = __v3 as u16;
}
loop {
if unlikely!(((unsafe { (*shost).sg_prot_tablesize }) as i32) < ((unsafe { (*shost).sg_tablesize }) as i32)) != 0 {
loop {
unsafe {
core::arch::asm!("683: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 683b - ., 3; .popsection\n# {0}", const 683 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_scsi_init_limits_8, sym __slate_asm_string_scsi_init_limits_9, const 2056 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe {
(*lim).max_integrity_segments = (*shost).sg_prot_tablesize;
}
}
unsafe {
(*lim).max_hw_sectors = (*shost).max_sectors;
}
unsafe {
(*lim).seg_boundary_mask = (*shost).dma_boundary;
}
unsafe {
(*lim).max_segment_size = (*shost).max_segment_size;
}
unsafe {
(*lim).virt_boundary_mask = (*shost).virt_boundary_mask;
}
unsafe {
(*lim).dma_alignment = max_t__2!(u32, unsafe { (*shost).dma_alignment }, dma_get_cache_alignment().wrapping_sub(1));
}
// Propagate the DMA formation properties to the dma-mapping layer as
// a courtesy service to the LLDDs.  This needs to check that the buses
// actually support the DMA API first, though.
if !unsafe { (*dev).dma_parms }.is_null() {
dma_set_seg_boundary(dev, unsafe { (*shost).dma_boundary });
dma_set_max_seg_size(dev, unsafe { (*shost).max_segment_size });
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct scsi_device_handler {
    list: list_head,
    module: *mut module,
    name: *const u8,
    check_sense: Option<unsafe extern "C" fn(*mut scsi_device, *mut scsi_sense_hdr) -> u32>,
    attach: Option<unsafe extern "C" fn(*mut scsi_device) -> i32>,
    detach: Option<unsafe extern "C" fn(*mut scsi_device)>,
    activate: Option<unsafe extern "C" fn(*mut scsi_device, Option<unsafe extern "C" fn(*mut (), i32)>, *mut ()) -> i32>,
    prep_fn: Option<unsafe extern "C" fn(*mut scsi_device, *mut request) -> u8>,
    set_params: Option<unsafe extern "C" fn(*mut scsi_device, *const u8) -> i32>,
    rescan: Option<unsafe extern "C" fn(*mut scsi_device)>,
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
fn __do_trace_scsi_dispatch_cmd_start(cmd: *mut scsi_cmnd) {
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
let mut __UNIQUE_ID_guard_612: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_612) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_613: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_613) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_615() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_scsi_dispatch_cmd_start)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_614: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_614;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut scsi_cmnd) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut scsi_cmnd) -> i32>>(__SCT__tp_func_scsi_dispatch_cmd_start as *const ()) };
unsafe { __v12.unwrap()(__data, cmd) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_scsi_dispatch_cmd_start: __SlateAlign16<[u8; 24]> = __SlateAlign16([115u8, 99u8, 115u8, 105u8, 95u8, 100u8, 105u8, 115u8, 112u8, 97u8, 116u8, 99u8, 104u8, 95u8, 99u8, 109u8, 100u8, 95u8, 115u8, 116u8, 97u8, 114u8, 116u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_scsi_dispatch_cmd_start_616: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_scsi_dispatch_cmd_start) }) as *mut ();

fn trace_scsi_dispatch_cmd_start(cmd: *mut scsi_cmnd) {
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_scsi_dispatch_cmd_start.key) }) != 0 {
__do_trace_scsi_dispatch_cmd_start(cmd);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_617: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_scsi_dispatch_cmd_error(cmd: *mut scsi_cmnd, rtn: i32) {
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
let mut __UNIQUE_ID_guard_619: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_619) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_620: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_620) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_622() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_scsi_dispatch_cmd_error)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_621: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_621;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut scsi_cmnd, i32) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut scsi_cmnd, i32) -> i32>>(__SCT__tp_func_scsi_dispatch_cmd_error as *const ()) };
unsafe { __v12.unwrap()(__data, cmd, rtn) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_scsi_dispatch_cmd_error: __SlateAlign16<[u8; 24]> = __SlateAlign16([115u8, 99u8, 115u8, 105u8, 95u8, 100u8, 105u8, 115u8, 112u8, 97u8, 116u8, 99u8, 104u8, 95u8, 99u8, 109u8, 100u8, 95u8, 101u8, 114u8, 114u8, 111u8, 114u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_scsi_dispatch_cmd_error_623: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_scsi_dispatch_cmd_error) }) as *mut ();

fn trace_scsi_dispatch_cmd_error(cmd: *mut scsi_cmnd, rtn: i32) {
if static_branch_unlikely__3!(unsafe { core::ptr::addr_of_mut!(__tracepoint_scsi_dispatch_cmd_error.key) }) != 0 {
__do_trace_scsi_dispatch_cmd_error(cmd, rtn);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_624: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn __do_trace_scsi_dispatch_cmd_done(cmd: *mut scsi_cmnd) {
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
let mut __UNIQUE_ID_guard_626: __SlateRecord_srcu_h_645 = class_srcu_fast_notrace_constructor(unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) });
let __slate_cleanup_ptr_7 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_626) as *mut _;
let __slate_cleanup_7 = ::slate_support::Guard(&mut || class_srcu_fast_notrace_destructor(__slate_cleanup_ptr_7 as *mut _));
let mut __UNIQUE_ID_unlock_627: *mut __SlateRecord_srcu_h_645 = ((unsafe { core::ptr::addr_of_mut!(tracepoint_srcu) }) as u64) as *mut __SlateRecord_srcu_h_645;
let __slate_cleanup_ptr_8 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_627) as *mut _;
let __slate_cleanup_8 = ::slate_support::Guard(&mut || __class_srcu_fast_notrace_cleanup_ctx(__slate_cleanup_ptr_8 as *mut _));
loop {
let mut __v9: *mut tracepoint_func;
let mut __v10: *mut tracepoint_func;
loop {
if !(__native_word__5!() || ((const { core::mem::size_of::<*mut tracepoint_func>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_629() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v10 = unsafe { core::ptr::read_volatile(core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!(__tracepoint_scsi_dispatch_cmd_done)).funcs) as *const *mut tracepoint_func) };
let __UNIQUE_ID_rcu_628: *mut tracepoint_func = __v10;
__v9 = __UNIQUE_ID_rcu_628;
let it_func_ptr: *mut tracepoint_func = __v9;
if !it_func_ptr.is_null() {
let __data: *mut () = unsafe { (*it_func_ptr).data };
let mut __v12: Option<unsafe extern "C" fn(*mut (), *mut scsi_cmnd) -> i32>;
__v12 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut (), *mut scsi_cmnd) -> i32>>(__SCT__tp_func_scsi_dispatch_cmd_done as *const ()) };
unsafe { __v12.unwrap()(__data, cmd) };
}
if true {
break;
}
}
}
}

#[used(compiler)]
#[unsafe(link_section = "__tracepoint_check")]
static __trace_check_scsi_dispatch_cmd_done: __SlateAlign16<[u8; 23]> = __SlateAlign16([115u8, 99u8, 115u8, 105u8, 95u8, 100u8, 105u8, 115u8, 112u8, 97u8, 116u8, 99u8, 104u8, 95u8, 99u8, 109u8, 100u8, 95u8, 100u8, 111u8, 110u8, 101u8, 0u8]);

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__tp_func_scsi_dispatch_cmd_done_630: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__tp_func_scsi_dispatch_cmd_done) }) as *mut ();

fn trace_scsi_dispatch_cmd_done(cmd: *mut scsi_cmnd) {
if static_branch_unlikely__2!(unsafe { core::ptr::addr_of_mut!(__tracepoint_scsi_dispatch_cmd_done.key) }) != 0 {
__do_trace_scsi_dispatch_cmd_done(cmd);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_631: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn scsi_ml_byte(result: i32) -> u8 {
(result >> 8i32 & 255) as u8
}

fn scsi_log_send(cmd: *mut scsi_cmnd) {
}

fn scsi_log_completion(cmd: *mut scsi_cmnd, disposition: i32) {
}

// scsi_init_limits()
// Size of integrity metadata is usually small, 1 inline sg should
// cover normal cases.
static mut scsi_sense_cache: *mut kmem_cache = unsafe { core::mem::zeroed() };

static mut scsi_sense_cache_mutex: mutex = {
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

#[unsafe(no_mangle)]
extern "C" fn scsi_init_sense_cache(shost: *mut Scsi_Host) -> i32 {
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!(scsi_sense_cache_mutex)) };
if unsafe { scsi_sense_cache }.is_null() {
unsafe {
scsi_sense_cache = kmem_cache_create_usercopy(c"scsi_sense_cache".as_ptr() as *const u8, 96u32, 0u32, __SLAB_FLAG_BIT!(4i32), 0u32, 96u32, None);
}
if unsafe { scsi_sense_cache }.is_null() {
ret = 12i32.wrapping_neg();
}
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!(scsi_sense_cache_mutex)) };
ret
}

fn scsi_set_blocked(cmd: *mut scsi_cmnd, reason: u32) {
let host: *mut Scsi_Host = unsafe { (*(*cmd).device).host };
let device: *mut scsi_device = unsafe { (*cmd).device };
let starget: *mut scsi_target = scsi_target(device);
// Set the appropriate busy bit for the device/host.
//
// If the host/device isn't busy, assume that something actually
// completed, and that we should be able to queue a command now.
//
// Note that the prior mid-layer assumption that any host could
// always queue at least one command is now broken.  The mid-layer
// will implement a user specifiable stall (see
// scsi_host.max_host_blocked and scsi_device.max_device_blocked)
// if a command is requeued with no other commands outstanding
// either for the device or for the host.
match reason as u32 {
4181 => {
atomic_set(unsafe { core::ptr::addr_of_mut!((*host).host_blocked) }, (unsafe { (*host).max_host_blocked }) as i32);
}
4182 | 4183 => {
atomic_set(unsafe { core::ptr::addr_of_mut!((*device).device_blocked) }, (unsafe { (*device).max_device_blocked }) as i32);
}
4184 => {
atomic_set(unsafe { core::ptr::addr_of_mut!((*starget).target_blocked) }, (unsafe { (*starget).max_target_blocked }) as i32);
}
_ => {
}
}
}

fn scsi_mq_requeue_cmd(cmd: *mut scsi_cmnd, msecs: u64) {
let rq: *mut request = scsi_cmd_to_rq(cmd);
if (unsafe { (*rq).rq_flags }) & 8u32 != 0 {
let __v2: u32 = (unsafe { (*rq).rq_flags }) & 4294967287u32;
unsafe {
(*rq).rq_flags = __v2;
}
scsi_mq_uninit_cmd(cmd);
} else {
let mut __v3: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("646: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 646b - ., 3; .popsection\n# {0}", const 646 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_mq_requeue_cmd_4, sym __slate_asm_string_scsi_mq_requeue_cmd_5, const 123 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("647: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 647b - ., 4; .popsection\n# {0}", const 647 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v3 = (__ret_warn_on != 0) as i64;
}
unsafe { blk_mq_requeue_request(rq, false) };
if scsi_host_in_recovery(unsafe { (*(*cmd).device).host }) == 0 {
unsafe { blk_mq_delay_kick_requeue_list((*rq).q, msecs) };
}
}

/// __scsi_queue_insert - private queue insertion
/// @cmd: The SCSI command being requeued
/// @reason:  The reason for the requeue
/// @unbusy: Whether the queue should be unbusied
///
/// This is a private queue insertion.  The public interface
/// scsi_queue_insert() always assumes the queue should be unbusied
/// because it's always called before the completion.  This function is
/// for a requeue after completion, which should only occur in this
/// file.
fn __scsi_queue_insert(cmd: *mut scsi_cmnd, reason: u32, unbusy: bool) {
let device: *mut scsi_device = unsafe { (*cmd).device };
SCSI_LOG_MLQUEUE!();
scsi_set_blocked(cmd, reason);
// Decrement the counters, since these commands are no longer
// active on the host/device.
if unbusy {
scsi_device_unbusy(device, cmd);
}
// Requeue this command.  It will go before all other commands
// that are already in the queue. Schedule requeue work under
// lock such that the kblockd_schedule_work() call happens
// before blk_mq_destroy_queue() finishes.
unsafe {
(*cmd).result = 0;
}
unsafe { blk_mq_requeue_request(scsi_cmd_to_rq(cmd), scsi_host_in_recovery((*(*cmd).device).host) == 0) };
}

/// scsi_queue_insert - Reinsert a command in the queue.
/// @cmd:    command that we are adding to queue.
/// @reason: why we are inserting command to queue.
///
/// We do this for one of two cases. Either the host is busy and it cannot accept
/// any more commands for the time being, or the device returned QUEUE_FULL and
/// can accept no more commands.
///
/// Context: This could be called either from an interrupt context or a normal
/// process context.
#[unsafe(no_mangle)]
extern "C" fn scsi_queue_insert(cmd: *mut scsi_cmnd, reason: u32) {
__scsi_queue_insert(cmd, reason, true);
}

/// scsi_failures_reset_retries - reset all failures to zero
/// @failures: &struct scsi_failures with specific failure modes set
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_failures_reset_retries")]
extern "C" fn scsi_failures_reset_retries(failures: *mut scsi_failures) {
let mut failure: *mut scsi_failure = unsafe { core::mem::zeroed() };
unsafe {
(*failures).total_retries = 0;
}
failure = unsafe { (*failures).failure_definitions };
while (unsafe { (*failure).result }) != 0 {
unsafe {
(*failure).retries = 0;
}
let __v1: *mut scsi_failure = failure;
failure = __v1.wrapping_offset(1 as isize);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_failures_reset_retries_648: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_failures)>, *mut ()>(Some(scsi_failures_reset_retries)) };

/// scsi_check_passthrough - Determine if passthrough scsi_cmnd needs a retry.
/// @scmd: scsi_cmnd to check.
/// @failures: scsi_failures struct that lists failures to check for.
///
/// Returns -EAGAIN if the caller should retry else 0.
fn scsi_check_passthrough(scmd: *mut scsi_cmnd, failures: *mut scsi_failures) -> i32 {
let mut __v3: bool = unsafe { core::mem::zeroed() };
let mut __v7: bool = unsafe { core::mem::zeroed() };
let mut __v10: bool = unsafe { core::mem::zeroed() };
let mut status: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut u32 = core::ptr::addr_of_mut!(status) as *mut u32;
let mut sshdr: core::mem::MaybeUninit<scsi_sense_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut scsi_sense_hdr = core::ptr::addr_of_mut!(sshdr) as *mut scsi_sense_hdr;
let mut failure: core::mem::MaybeUninit<*mut scsi_failure> = core::mem::MaybeUninit::uninit();
let __slate_slot_13: *mut *mut scsi_failure = core::ptr::addr_of_mut!(failure) as *mut *mut scsi_failure;
unsafe {
let __slate_auto_bytes = __slate_slot_13 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if (*scmd).result == 0 {
return 0;
} else {
if failures.is_null() {
return 0;
} else {
*__slate_slot_13 = (*failures).failure_definitions;
'__join_11: {
while (*(*__slate_slot_13)).result != 0 {
if (*(*__slate_slot_13)).result == 2147483647 {
break '__join_11;
} else {
if host_byte!(unsafe { (*scmd).result }) != 0 && host_byte!(unsafe { (*scmd).result }) == host_byte!(unsafe { (*(*__slate_slot_13)).result }) {
break '__join_11;
} else {
*__slate_slot_11 = status_byte!(unsafe { (*scmd).result }) as u32;
if (*__slate_slot_11 as u32) != 0 {
if (*(*__slate_slot_13)).result == 255 {
__v10 = !scsi_status_is_good((*scmd).result);
} else {
__v10 = false;
}
if __v10 {
break '__join_11;
} else {
if (*__slate_slot_11 as u32) == (status_byte!(unsafe { (*(*__slate_slot_13)).result }) as u32) {
if status_byte!(unsafe { (*(*__slate_slot_13)).result }) != 2 || ((*(*__slate_slot_13)).sense as i32) == 255 {
break '__join_11;
} else {
if !scsi_command_normalize_sense(scmd as *const scsi_cmnd, core::ptr::addr_of_mut!(*__slate_slot_12)) {
return 0;
} else {
if ((*(*__slate_slot_13)).sense as i32) == ((*__slate_slot_12).sense_key as i32) {
if ((*(*__slate_slot_13)).asc as i32) == 255 {
break '__join_11;
} else {
if ((*(*__slate_slot_13)).asc as i32) == ((*__slate_slot_12).asc as i32) {
if ((*(*__slate_slot_13)).ascq as i32) == 255 || ((*(*__slate_slot_13)).ascq as i32) == ((*__slate_slot_12).ascq as i32) {
break '__join_11;
}
}
}
}
}
}
}
}
}
let __v9: *mut scsi_failure = *__slate_slot_13;
*__slate_slot_13 = __v9.wrapping_offset(1 as isize);
}
}
}
return 0;
}
if (*(*__slate_slot_13)).allowed != 0 {
if ((*(*__slate_slot_13)).allowed as i32) == 1i32.wrapping_neg() {
__v3 = true;
} else {
let __v2: *mut scsi_failure = *__slate_slot_13;
let __v0: i8 = ((*__v2).retries as i32).wrapping_add(1) as i8;
(*__v2).retries = __v0;
__v3 = (__v0 as i32) <= ((*(*__slate_slot_13)).allowed as i32);
}
if __v3 {
return 11i32.wrapping_neg();
}
} else {
if (*failures).total_allowed == 1i32.wrapping_neg() {
__v7 = true;
} else {
let __v5: i32 = (*failures).total_retries;
let __v4: i32 = __v5.wrapping_add(1);
(*failures).total_retries = __v4;
__v7 = __v4 <= (*failures).total_allowed;
}
if __v7 {
return 11i32.wrapping_neg();
}
}
return 0;
}
}
}
}

/// scsi_execute_cmd - insert request and wait for the result
/// @sdev:	scsi_device
/// @cmd:	scsi command
/// @opf:	block layer request cmd_flags
/// @buffer:	data buffer
/// @bufflen:	len of buffer
/// @timeout:	request timeout in HZ
/// @ml_retries:	number of times SCSI midlayer will retry request
/// @args:	Optional args. See struct definition for field descriptions
///
/// Returns the scsi_cmnd result field if a command was executed, or a negative
/// Linux error code if we didn't get that far.
#[unsafe(no_mangle)]
extern "C" fn scsi_execute_cmd(sdev: *mut scsi_device, cmd: *const u8, opf: u32, buffer: *mut (), bufflen: u32, timeout: i32, ml_retries: i32, mut args: *const scsi_exec_args) -> i32 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut scmd: core::mem::MaybeUninit<*mut scsi_cmnd> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut *mut scsi_cmnd = core::ptr::addr_of_mut!(scmd) as *mut *mut scsi_cmnd;
let mut req: core::mem::MaybeUninit<*mut request> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut request = core::ptr::addr_of_mut!(req) as *mut *mut request;
unsafe {
let __slate_auto_bytes = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_1 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
let __slate_auto_bytes = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if args.is_null() {
args = core::ptr::addr_of!(default_args.0);
} else {
__ret_warn_on = (!unsafe { (*args).sense }.is_null() && (*args).sense_len != 96) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("649: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 649b - ., 3; .popsection\n# {0}", const 649 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_execute_cmd_12, sym __slate_asm_string_scsi_execute_cmd_13, const 309 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("650: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 650b - ., 4; .popsection\n# {0}", const 650 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
if __ret_warn_on != 0 {
return 22i32.wrapping_neg();
}
}
'__join_0: {
loop {
*__slate_slot_11 = scsi_alloc_request((*sdev).request_queue, opf, (*args).req_flags);
if IS_ERR(*__slate_slot_11 as *const ()) {
return PTR_ERR(*__slate_slot_11 as *const ()) as i32;
} else {
if bufflen != 0 {
*__slate_slot_9 = blk_rq_map_kern(*__slate_slot_11, buffer, bufflen, (BIT!(10i32) | BIT!(11i32)) as u32);
if *__slate_slot_9 != 0 {
break '__join_0;
}
}
*__slate_slot_10 = blk_mq_rq_to_pdu(*__slate_slot_11) as *mut scsi_cmnd;
(*(*__slate_slot_10)).cmd_len = COMMAND_SIZE!(unsafe { *cmd.wrapping_offset(0 as isize) }) as u16;
memcpy(((*(*__slate_slot_10)).cmnd.as_mut_ptr() as *mut u8) as *mut (), cmd as *const (), (*(*__slate_slot_10)).cmd_len as u64);
(*(*__slate_slot_10)).allowed = ml_retries;
let __v5: *mut scsi_cmnd = *__slate_slot_10;
(*__v5).flags = (*__v5).flags | (*args).scmd_flags;
(*(*__slate_slot_11)).timeout = timeout as u32;
let __v2: *mut request = *__slate_slot_11;
(*__v2).rq_flags = (*__v2).rq_flags | 128u32;
// head injection *required* here otherwise quiesce won't work
blk_execute_rq(*__slate_slot_11, true);
if scsi_check_passthrough(*__slate_slot_10, (*args).failures) == 11i32.wrapping_neg() {
blk_mq_free_request(*__slate_slot_11);
} else {
break;
}
}
}
// Some devices (USB mass-storage in particular) may transfer
// garbage data together with a residue indicating that the data
// is invalid.  Prevent the garbage from being misinterpreted
// and prevent security leaks by zeroing out the excess data.
if unlikely!((*(*__slate_slot_10)).resid_len > 0 && (*(*__slate_slot_10)).resid_len <= bufflen) != 0 {
memset((((buffer as *const u8).wrapping_offset(bufflen as isize) as *mut ()) as *const u8).wrapping_offset(-((*(*__slate_slot_10)).resid_len as isize)) as *mut (), 0i32, (*(*__slate_slot_10)).resid_len as u64);
}
if !unsafe { (*args).resid }.is_null() {
*(*args).resid = (*(*__slate_slot_10)).resid_len as i32;
}
if !unsafe { (*args).sense }.is_null() {
memcpy((*args).sense as *mut (), (*(*__slate_slot_10)).sense_buffer as *const (), 96u64);
}
if !unsafe { (*args).sshdr }.is_null() {
scsi_normalize_sense((*(*__slate_slot_10)).sense_buffer as *const u8, (*(*__slate_slot_10)).sense_len as i32, (*args).sshdr);
}
*__slate_slot_9 = (*(*__slate_slot_10)).result;
}
blk_mq_free_request(*__slate_slot_11);
return *__slate_slot_9;
}
}

static default_args: __SlateConst<scsi_exec_args> = __SlateConst(unsafe { core::mem::zeroed() });

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_execute_cmd_651: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, *const u8, u32, *mut (), u32, i32, i32, *const scsi_exec_args) -> i32>, *mut ()>(Some(scsi_execute_cmd)) };

/// Wake up the error handler if necessary. Avoid as follows that the error
/// handler is not woken up if host in-flight requests number ==
/// shost->host_failed: use call_rcu() in scsi_eh_scmd_add() in combination
/// with an RCU read lock in this function to ensure that this function in
/// its entirety either finishes before scsi_eh_scmd_add() increases the
/// host_failed counter or that it notices the shost state change made by
/// scsi_eh_scmd_add().
fn scsi_dec_host_busy(shost: *mut Scsi_Host, cmd: *mut scsi_cmnd) {
let mut flags: u64 = 0;
rcu_read_lock();
__clear_bit!(1i32, unsafe { core::ptr::addr_of_mut!((*cmd).state) });
if unlikely__2!(scsi_host_in_recovery(shost)) != 0 {
// Ensure the clear of SCMD_STATE_INFLIGHT is visible to
// other CPUs before counting busy requests. Otherwise,
// reordering can cause CPUs to race and miss an eh wakeup
// when no CPU sees all busy requests as done or timed out.
loop {
kcsan_mb!();
unsafe {
core::arch::asm!("lock addl $0,-4(%rsp)", options(nostack, att_syntax));
}
if true {
break;
}
}
let busy: u32 = (unsafe { scsi_host_busy(shost) }) as u32;
spin_lock_irqsave!(unsafe { (*shost).host_lock }, flags);
if (unsafe { (*shost).host_failed }) != 0 || (unsafe { (*shost).host_eh_scheduled }) != 0 {
unsafe { scsi_eh_wakeup(shost, busy) };
}
spin_unlock_irqrestore(unsafe { (*shost).host_lock }, flags);
}
rcu_read_unlock();
}

#[unsafe(no_mangle)]
extern "C" fn scsi_device_unbusy(sdev: *mut scsi_device, cmd: *mut scsi_cmnd) {
let shost: *mut Scsi_Host = unsafe { (*sdev).host };
let starget: *mut scsi_target = scsi_target(sdev);
scsi_dec_host_busy(shost, cmd);
if (unsafe { (*starget).can_queue }) > 0 {
atomic_dec(unsafe { core::ptr::addr_of_mut!((*starget).target_busy) });
}
if !unsafe { (*sdev).budget_map.map }.is_null() {
sbitmap_put(unsafe { core::ptr::addr_of_mut!((*sdev).budget_map) }, (unsafe { (*cmd).budget_token }) as u32);
}
unsafe {
(*cmd).budget_token = 1i32.wrapping_neg();
}
}

/// Kick the queue of SCSI device @sdev if @sdev != current_sdev. Called with
/// interrupts disabled.
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_kick_sdev_queue")]
extern "C" fn scsi_kick_sdev_queue(sdev: *mut scsi_device, data: *mut ()) {
let current_sdev: *mut scsi_device = data as *mut scsi_device;
if sdev != current_sdev {
unsafe { blk_mq_run_hw_queues((*sdev).request_queue, true) };
}
}

/// Called for single_lun devices on IO completion. Clear starget_sdev_user,
/// and call blk_run_queue for all the scsi_devices on the target -
/// including current_sdev first.
///
/// Called with *no* scsi locks held.
fn scsi_single_lun_run(current_sdev: *mut scsi_device) {
let shost: *mut Scsi_Host = unsafe { (*current_sdev).host };
let starget: *mut scsi_target = scsi_target(current_sdev);
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { (*shost).host_lock }, flags);
unsafe {
(*starget).starget_sdev_user = core::ptr::null_mut::<scsi_device>();
}
spin_unlock_irqrestore(unsafe { (*shost).host_lock }, flags);
// Call blk_run_queue for all LUNs on the target, starting with
// current_sdev. We race with others (to set starget_sdev_user),
// but in most cases, we will be first. Ideally, each LU on the
// target would get some limited time or requests on the target.
unsafe { blk_mq_run_hw_queues((*current_sdev).request_queue, ((*shost).__slate_bits_0.__get_queuecommand_may_block() as i32) != 0) };
spin_lock_irqsave!(unsafe { (*shost).host_lock }, flags);
if unsafe { (*starget).starget_sdev_user }.is_null() {
unsafe { __starget_for_each_device(starget, current_sdev as *mut (), Some(scsi_kick_sdev_queue)) };
}
spin_unlock_irqrestore(unsafe { (*shost).host_lock }, flags);
}

fn scsi_device_is_busy(sdev: *mut scsi_device) -> bool {
if scsi_device_busy(sdev) >= ((unsafe { (*sdev).queue_depth }) as i32) {
return true;
}
if atomic_read((unsafe { core::ptr::addr_of_mut!((*sdev).device_blocked) }) as *const __SlateRecord_types_h_190) > 0 {
return true;
}
false
}

fn scsi_target_is_busy(starget: *mut scsi_target) -> bool {
if (unsafe { (*starget).can_queue }) > 0 {
if (atomic_read((unsafe { core::ptr::addr_of_mut!((*starget).target_busy) }) as *const __SlateRecord_types_h_190) as u32) >= unsafe { (*starget).can_queue } {
return true;
}
if atomic_read((unsafe { core::ptr::addr_of_mut!((*starget).target_blocked) }) as *const __SlateRecord_types_h_190) > 0 {
return true;
}
}
false
}

fn scsi_host_is_busy(shost: *mut Scsi_Host) -> bool {
if atomic_read((unsafe { core::ptr::addr_of_mut!((*shost).host_blocked) }) as *const __SlateRecord_types_h_190) > 0 {
return true;
}
if ((unsafe { (*shost).__slate_bits_0.__get_host_self_blocked() }) as i32) != 0 {
return true;
}
false
}

fn scsi_starved_list_run(shost: *mut Scsi_Host) {
let mut starved_list: list_head = unsafe { core::mem::zeroed() };
starved_list = list_head { next: core::ptr::addr_of_mut!(starved_list), prev: core::ptr::addr_of_mut!(starved_list) };
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { (*shost).host_lock }, flags);
list_splice_init(unsafe { core::ptr::addr_of_mut!((*shost).starved_list) }, core::ptr::addr_of_mut!(starved_list));
while list_empty(core::ptr::addr_of_mut!(starved_list) as *const list_head) == 0 {
// As long as shost is accepting commands and we have
// starved queues, call blk_run_queue. scsi_request_fn
// drops the queue_lock and can add us back to the
// starved_list.
//
// host_lock protects the starved_list and starved_entry.
// scsi_request_fn must get the host_lock before checking
// or modifying starved_list or starved_entry.
if scsi_host_is_busy(shost) {
break;
}
sdev = list_entry__15!(starved_list.next, scsi_device);
list_del_init(unsafe { core::ptr::addr_of_mut!((*sdev).starved_entry) });
if scsi_target_is_busy(scsi_target(sdev)) {
list_move_tail(unsafe { core::ptr::addr_of_mut!((*sdev).starved_entry) }, unsafe { core::ptr::addr_of_mut!((*shost).starved_list) });
} else {
// Once we drop the host lock, a racing scsi_remove_device()
// call may remove the sdev from the starved list and destroy
// it and the queue.  Mitigate by taking a reference to the
// queue and never touching the sdev again after we drop the
// host lock.  Note: if __scsi_remove_device() invokes
// blk_mq_destroy_queue() before the queue is run from this
// function then blk_run_queue() will return immediately since
// blk_mq_destroy_queue() marks the queue with QUEUE_FLAG_DYING.
let slq: *mut request_queue = unsafe { (*sdev).request_queue };
if unsafe { blk_get_queue(slq) } {
spin_unlock_irqrestore(unsafe { (*shost).host_lock }, flags);
unsafe { blk_mq_run_hw_queues(slq, false) };
unsafe { blk_put_queue(slq) };
spin_lock_irqsave!(unsafe { (*shost).host_lock }, flags);
}
}
}
// put any unprocessed entries back
list_splice(core::ptr::addr_of_mut!(starved_list) as *const list_head, unsafe { core::ptr::addr_of_mut!((*shost).starved_list) });
spin_unlock_irqrestore(unsafe { (*shost).host_lock }, flags);
}

/// scsi_run_queue - Select a proper request queue to serve next.
/// @q:  last request's queue
///
/// The previous command was completely finished, start a new one if possible.
fn scsi_run_queue(q: *mut request_queue) {
let sdev: *mut scsi_device = (unsafe { (*q).queuedata }) as *mut scsi_device;
if ((unsafe { (*scsi_target(sdev)).__slate_bits_0.__get_single_lun() }) as i32) != 0 {
scsi_single_lun_run(sdev);
}
if list_empty((unsafe { core::ptr::addr_of_mut!((*(*sdev).host).starved_list) }) as *const list_head) == 0 {
scsi_starved_list_run(unsafe { (*sdev).host });
}
// Note: blk_mq_kick_requeue_list() runs the queue asynchronously.
unsafe { blk_mq_kick_requeue_list(q) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_requeue_run_queue")]
extern "C" fn scsi_requeue_run_queue(work: *mut work_struct) {
let sdev: *mut scsi_device = container_of__20!(work, scsi_device);
let q: *mut request_queue = unsafe { (*sdev).request_queue };
scsi_run_queue(q);
}

#[unsafe(no_mangle)]
extern "C" fn scsi_run_host_queues(shost: *mut Scsi_Host) {
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut prev: *mut scsi_device = core::ptr::null_mut::<scsi_device>();
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { (*shost).host_lock }, flags);
sdev = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*shost).__devices) });
while list_entry_is_head!(sdev, unsafe { core::ptr::addr_of_mut!((*shost).__devices) }, siblings) == 0 {
// Only skip devices so deep into removal they will never need
// another kick to their queues. Thus scsi_device_get() cannot
// be used as it would skip devices in SDEV_CANCEL state which
// may need a queue kick.
let __v4: bool = ((unsafe { (*sdev).sdev_state }) as u32) == 4 || unsafe { get_device(core::ptr::addr_of_mut!((*sdev).sdev_gendev)) }.is_null();
if !__v4 {
spin_unlock_irqrestore(unsafe { (*shost).host_lock }, flags);
if !prev.is_null() {
unsafe { put_device(core::ptr::addr_of_mut!((*prev).sdev_gendev)) };
}
scsi_run_queue(unsafe { (*sdev).request_queue });
prev = sdev;
spin_lock_irqsave!(unsafe { (*shost).host_lock }, flags);
}
sdev = list_next_entry__2!(sdev, siblings);
}
spin_unlock_irqrestore(unsafe { (*shost).host_lock }, flags);
if !prev.is_null() {
unsafe { put_device(core::ptr::addr_of_mut!((*prev).sdev_gendev)) };
}
}

fn scsi_uninit_cmd(cmd: *mut scsi_cmnd) {
if !blk_rq_is_passthrough(scsi_cmd_to_rq(cmd)) {
let drv: *mut scsi_driver = scsi_cmd_to_driver(cmd);
if (unsafe { (*drv).uninit_command }) != None {
unsafe { unsafe { (*drv).uninit_command }.unwrap()(cmd) };
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_free_sgtables")]
extern "C" fn scsi_free_sgtables(cmd: *mut scsi_cmnd) {
if (unsafe { (*cmd).sdb.table.nents }) != 0 {
unsafe { sg_free_table_chained(core::ptr::addr_of_mut!((*cmd).sdb.table), 2u32) };
}
if scsi_prot_sg_count(cmd) != 0 {
unsafe { sg_free_table_chained(core::ptr::addr_of_mut!((*(*cmd).prot_sdb).table), 1u32) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_free_sgtables_652: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_cmnd)>, *mut ()>(Some(scsi_free_sgtables)) };

fn scsi_mq_uninit_cmd(cmd: *mut scsi_cmnd) {
scsi_free_sgtables(cmd);
scsi_uninit_cmd(cmd);
}

fn scsi_run_queue_async(sdev: *mut scsi_device) {
if scsi_host_in_recovery(unsafe { (*sdev).host }) != 0 {
return;
}
let __v0: bool = ((unsafe { (*scsi_target(sdev)).__slate_bits_0.__get_single_lun() }) as i32) != 0 || list_empty((unsafe { core::ptr::addr_of_mut!((*(*sdev).host).starved_list) }) as *const list_head) == 0;
if __v0 {
unsafe { kblockd_schedule_work(core::ptr::addr_of_mut!((*sdev).requeue_work)) };
} else {
// smp_mb() present in sbitmap_queue_clear() or implied in
// .end_io is for ordering writing .device_busy in
// scsi_device_unbusy() and reading sdev->restarts.
let old: i32 = atomic_read((unsafe { core::ptr::addr_of_mut!((*sdev).restarts) }) as *const __SlateRecord_types_h_190);
// ->restarts has to be kept as non-zero if new budget
//  contention occurs.
//
//  No need to run queue when either another re-run
//  queue wins in updating ->restarts or a new budget
//  contention occurs.
let __v1: bool = old != 0 && atomic_cmpxchg(unsafe { core::ptr::addr_of_mut!((*sdev).restarts) }, old, 0i32) == old;
if __v1 {
unsafe { blk_mq_run_hw_queues((*sdev).request_queue, true) };
}
}
}

/// Returns false when no more bytes to process, true if there are more
fn scsi_end_request(req: *mut request, error: u8, bytes: u32) -> bool {
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(req) as *mut scsi_cmnd;
let sdev: *mut scsi_device = unsafe { (*cmd).device };
let q: *mut request_queue = unsafe { (*sdev).request_queue };
if unsafe { blk_update_request(req, error, bytes) } {
return true;
}
if (unsafe { (*q).limits.features }) & 8u32 != 0 {
unsafe { add_disk_randomness((*(*req).q).disk) };
}
let mut __v0: i64;
let __ret_warn_on: i32 = (!blk_rq_is_passthrough(req) && (unsafe { (*cmd).flags }) & 2i32 == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("653: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 653b - ., 3; .popsection\n# {0}", const 653 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_end_request_1, sym __slate_asm_string_scsi_end_request_2, const 678 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("654: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 654b - ., 4; .popsection\n# {0}", const 654 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe {
(*cmd).flags = 0;
}
// Calling rcu_barrier() is not necessary here because the
// SCSI error handler guarantees that the function called by
// call_rcu() has been called before scsi_end_request() is
// called.
destroy_rcu_head(unsafe { core::ptr::addr_of_mut!((*cmd).rcu) });
// In the MQ case the command gets freed by __blk_mq_end_request,
// so we have to do all cleanup that depends on it earlier.
//
// We also can't kick the queues from irq context, so we
// will have to defer it to a workqueue.
scsi_mq_uninit_cmd(cmd);
// queue is still alive, so grab the ref for preventing it
// from being cleaned up during running queue.
percpu_ref_get(unsafe { core::ptr::addr_of_mut!((*q).q_usage_counter) });
unsafe { __blk_mq_end_request(req, error) };
scsi_run_queue_async(sdev);
percpu_ref_put(unsafe { core::ptr::addr_of_mut!((*q).q_usage_counter) });
false
}

/// scsi_result_to_blk_status - translate a SCSI result code into blk_status_t
/// @result:	scsi error code
///
/// Translate a SCSI result code into a blk_status_t value.
fn scsi_result_to_blk_status(result: i32) -> u8 {
// Check the scsi-ml byte first in case we converted a host or status
// byte.
match scsi_ml_byte(result) as i32 {
0 => {
}
1 => {
return 6;
}
2 => {
return 3;
}
3 => {
return 7;
}
4 => {
return 5;
}
5 => {
return 17;
}
_ => {
}
}
match host_byte!(result) {
0 => {
if scsi_status_is_good(result) {
return 0;
}
return 10;
}
15 | 20 => {
return 4;
}
_ => {
return 10;
}
}
unsafe { core::mem::zeroed() }
}

/// scsi_rq_err_bytes - determine number of bytes till the next failure boundary
/// @rq: request to examine
///
/// Description:
///     A request could be merge of IOs which require different failure
///     handling.  This function determines the number of bytes which
///     can be failed from the beginning of the request without
///     crossing into area which need to be retried further.
///
/// Return:
///     The number of bytes to fail.
fn scsi_rq_err_bytes(rq: *const request) -> u32 {
let ff: u32 = (unsafe { (*rq).cmd_flags }) & 1792u32;
let mut bytes: u32 = 0;
let mut bio: *mut bio = unsafe { core::mem::zeroed() };
if (unsafe { (*rq).rq_flags }) & 4u32 == 0 {
return blk_rq_bytes(rq);
}
// Currently the only 'mixing' which can happen is between
// different fastfail types.  We can safely fail portions
// which have all the failfast bits that the first one has -
// the ones which are at least as eager to fail as the first
// one.
bio = unsafe { (*rq).bio };
while !bio.is_null() {
if (unsafe { (*bio).bi_opf }) & ff != ff {
break;
}
bytes = bytes.wrapping_add(unsafe { (*bio).bi_iter.bi_size });
bio = unsafe { (*bio).bi_next };
}
// this could lead to infinite loop
loop {
if unlikely!(blk_rq_bytes(rq) != 0 && bytes == 0) != 0 {
loop {
unsafe {
core::arch::asm!("655: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 655b - ., 3; .popsection\n# {0}", const 655 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_scsi_rq_err_bytes_3, sym __slate_asm_string_scsi_rq_err_bytes_4, const 788 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
bytes
}

fn scsi_cmd_runtime_exceeced(cmd: *mut scsi_cmnd) -> bool {
let req: *mut request = scsi_cmd_to_rq(cmd);
let mut wait_for: u64 = 0;
if (unsafe { (*cmd).allowed }) == 1i32.wrapping_neg() {
return false;
}
wait_for = (unsafe { (*cmd).allowed }.wrapping_add(1) as u32).wrapping_mul(unsafe { (*req).timeout }) as u64;
if time_before__2!(unsafe { (*cmd).jiffies_at_alloc }.wrapping_add(wait_for), unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }) {
unsafe { scmd_printk(c"\x013".as_ptr() as *const u8, cmd, c"timing out command, waited %lus\n".as_ptr() as *const u8, wait_for / 1000) };
return true;
}
false
}

// When ALUA transition state is returned, reprep the cmd to
// use the ALUA handler's transition timeout. Delay the reprep
// 1 sec to avoid aggressive retries of the target in that
// state.
/// Helper for scsi_io_completion() when special action required.
fn scsi_io_completion_action(cmd: *mut scsi_cmnd, result: i32) {
let req: *mut request = scsi_cmd_to_rq(cmd);
let mut level: i32 = 0;
let mut action: u32 = 0;
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut sense_valid: bool = false;
let mut sense_current: bool = true; // false implies "deferred sense"
let mut blk_stat: u8 = 0;
sense_valid = unsafe { scsi_command_normalize_sense(cmd as *const scsi_cmnd, core::ptr::addr_of_mut!(sshdr)) };
if sense_valid {
sense_current = !scsi_sense_is_deferred(unsafe { &(*(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr)) });
}
blk_stat = scsi_result_to_blk_status(result);
if host_byte!(result) == 8 {
// Third party bus reset or reset for error recovery
// reasons.  Just retry the command and see what
// happens.
action = 3;
} else {
if sense_valid && sense_current {
match sshdr.sense_key as i32 {
6 => {
if ((unsafe { (*(*cmd).device).__slate_bits_0.__get_removable() }) as i32) != 0 {
// Detected disc change.  Set a bit
// and quietly refuse further access.
unsafe {
(*(*cmd).device).__slate_bits_0.__set_changed(1u32);
}
action = 0;
} else {
// Must have been a power glitch, or a
// bus reset.  Could not have been a
// media change, so we just retry the
// command and see what happens.
action = 3;
}
}
5 => {
if ((unsafe { (*(*cmd).device).__slate_bits_0.__get_use_10_for_rw() }) as i32) != 0 && (sshdr.asc as i32) == 32 && (sshdr.ascq as i32) == 0 && (((unsafe { *((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) == 40 || ((unsafe { *((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) }) as i32) == 42) {
// This will issue a new 6-byte command.
unsafe {
(*(*cmd).device).__slate_bits_0.__set_use_10_for_rw(0u32);
}
action = 1;
} else {
if (sshdr.asc as i32) == 16 {
// DIX
action = 0;
blk_stat = 8;
// INVALID COMMAND OPCODE or INVALID FIELD IN CDB
} else {
if (sshdr.asc as i32) == 32 || (sshdr.asc as i32) == 36 {
action = 0;
blk_stat = 5;
} else {
action = 0;
}
}
}
// If we had an ILLEGAL REQUEST returned, then
// we may have performed an unsupported
// command.  The only thing this should be
// would be a ten byte read where only a six
// byte read was supported.  Also, on a system
// where READ CAPACITY failed, we may have
// read past the end of the disk.
}
11 => {
action = 0;
if (sshdr.asc as i32) == 16 {
blk_stat = 8;
}
// DIF
}
2 => {
if (sshdr.asc as i32) == 4 {
'__slate_break_1: {
match sshdr.ascq as i32 {
1 | 4..=9 | 17 | 20 | 26 | 27 | 29 => {
action = 4; // becoming ready
// format in progress
// rebuild in progress
// recalculation in progress
// operation in progress
// Long write in progress
// self test in progress
// notify (enable spinup) required
// space allocation in progress
// start stop unit in progress
// sanitize in progress
// configuration in progress
}
10 => {
action = 2; // ALUA state transition
break '__slate_break_1;
// Depopulation might take many hours,
// thus it is not worthwhile to retry.
}
_ => {
// depopulation in progress
// depopulation restore in progress
action = 0;
}
}
}
} else {
action = 0;
}
// If the device is in the process of becoming
// ready, or has a temporary blockage, retry.
}
13 => {
action = 0; // See SSC3rXX or current.
}
7 => {
action = 0;
if (sshdr.asc as i32) == 12 && (sshdr.ascq as i32) == 18 || (sshdr.asc as i32) == 85 && ((sshdr.ascq as i32) == 14 || (sshdr.ascq as i32) == 15) {
// Insufficient zone resources
blk_stat = 14;
}
}
_ => {
action = 0;
}
}
} else {
action = 0;
}
}
let __v2: bool = (action as u32) != 0 && scsi_cmd_runtime_exceeced(cmd);
if __v2 {
action = 0;
}
// if logging is enabled the failure will be printed
// in scsi_log_completion(), so avoid duplicate messages
// Give up and fail the remainder of the request
// Retry the same command immediately
// Retry the same command after a delay
match action as u32 {
0 => {
if (unsafe { (*req).rq_flags }) & 128u32 == 0 {
if unlikely__3!(unsafe { scsi_logging_level }) != 0 {
level = SCSI_LOG_LEVEL!();
}
// if logging is enabled the failure will be printed
// in scsi_log_completion(), so avoid duplicate messages
let __v4: bool = level == 0 && __ratelimit__2!(unsafe { core::ptr::addr_of_mut!(_rs) }) != 0;
if __v4 {
unsafe { scsi_print_result(cmd, core::ptr::null::<u8>(), 8195i32) };
if sense_valid {
unsafe { scsi_print_sense(cmd) };
}
unsafe { scsi_print_command(cmd) };
}
}
// Give up and fail the remainder of the request
if !scsi_end_request(req, blk_stat, scsi_rq_err_bytes(req as *const request)) {
return;
}
scsi_mq_requeue_cmd(cmd, 0u64);
}
1 => {
scsi_mq_requeue_cmd(cmd, 0u64);
}
2 => {
scsi_mq_requeue_cmd(cmd, 1000u64);
}
3 => {
__scsi_queue_insert(cmd, 4183u32, false); // Retry the same command immediately
}
4 => {
__scsi_queue_insert(cmd, 4182u32, false); // Retry the same command after a delay
}
_ => {
}
}
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

// Helper for scsi_io_completion() when cmd->result is non-zero. Returns a
// new result that may suppress further error checking. Also modifies
// *blk_statp in some cases.
fn scsi_io_completion_nz_result(cmd: *mut scsi_cmnd, mut result: i32, blk_statp: &mut u8) -> i32 {
let mut sense_valid: bool = false;
let mut sense_current: bool = true; // false implies "deferred sense"
let req: *mut request = scsi_cmd_to_rq(cmd);
let mut sshdr: scsi_sense_hdr = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(sshdr) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
sense_valid = unsafe { scsi_command_normalize_sense(cmd as *const scsi_cmnd, core::ptr::addr_of_mut!(sshdr)) };
if sense_valid {
sense_current = !scsi_sense_is_deferred(unsafe { &(*(core::ptr::addr_of_mut!(sshdr) as *const scsi_sense_hdr)) });
}
if blk_rq_is_passthrough(req) {
if sense_valid {
// SG_IO wants current and deferred errors
let mut __v0: i32;
let __UNIQUE_ID_x__656: i32 = 8i32.wrapping_add((unsafe { *unsafe { (*cmd).sense_buffer }.wrapping_offset(7 as isize) }) as i32);
let __UNIQUE_ID_y__657: i32 = 96;
loop {
if __types_ok__2!(__UNIQUE_ID_x__656, __UNIQUE_ID_y__657) == 0 {
unsafe { __compiletime_assert_658() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__656 < __UNIQUE_ID_y__657 { __UNIQUE_ID_x__656 } else { __UNIQUE_ID_y__657 };
unsafe {
(*cmd).sense_len = __v0 as u32;
}
}
if sense_current {
unsafe {
*(blk_statp as *mut u8) = scsi_result_to_blk_status(result);
}
}
} else {
if blk_rq_bytes(req as *const request) == 0 && sense_current {
// Flush commands do not transfers any data, and thus cannot use
// good_bytes != blk_rq_bytes(req) as the signal for an error.
// This sets *blk_statp explicitly for the problem case.
unsafe {
*(blk_statp as *mut u8) = scsi_result_to_blk_status(result);
}
}
}
// Recovered errors need reporting, but they're always treated as
// success, so fiddle the result code here.  For passthrough requests
// we already took a copy of the original into sreq->result which
// is what gets returned to the user
if sense_valid && (sshdr.sense_key as i32) == 1 {
let mut do_print: bool = true;
// if ATA PASS-THROUGH INFORMATION AVAILABLE [0x0, 0x1d]
// skip print since caller wants ATA registers. Only occurs
// on SCSI ATA PASS_THROUGH commands when CK_COND=1
if (sshdr.asc as i32) == 0 && (sshdr.ascq as i32) == 29 {
do_print = false;
} else {
if (unsafe { (*req).rq_flags }) & 128u32 != 0 {
do_print = false;
}
}
if do_print {
unsafe { scsi_print_sense(cmd) };
}
result = 0;
// for passthrough, *blk_statp may be set
unsafe {
*(blk_statp as *mut u8) = 0;
}
}
// Another corner case: the SCSI status byte is non-zero but 'good'.
// Example: PRE-FETCH command returns SAM_STAT_CONDITION_MET when
// it is able to fit nominated LBs in its cache (and SAM_STAT_GOOD
// if it can't fit). Treat SAM_STAT_CONDITION_MET and the related
// intermediate statuses (both obsolete in SAM-4) as good.
let __v2: bool = result & 255 != 0 && scsi_status_is_good(result);
if __v2 {
result = 0;
unsafe {
*(blk_statp as *mut u8) = 0;
}
}
result
}

/// scsi_io_completion - Completion processing for SCSI commands.
/// @cmd:	command that is finished.
/// @good_bytes:	number of processed bytes.
///
/// We will finish off the specified number of sectors. If we are done, the
/// command block will be released and the queue function will be goosed. If we
/// are not done then we have to figure out what to do next:
///
///   a) We can call scsi_mq_requeue_cmd().  The request will be
/// 	unprepared and put back on the queue.  Then a new command will
/// 	be created for it.  This should be used if we made forward
/// 	progress, or if we want to switch from READ(10) to READ(6) for
/// 	example.
///
///   b) We can call scsi_io_completion_action().  The request will be
/// 	put back on the queue and retried using the same command as
/// 	before, possibly after a delay.
///
///   c) We can call scsi_end_request() with blk_stat other than
/// 	BLK_STS_OK, to fail the remainder of the request.
#[unsafe(no_mangle)]
extern "C" fn scsi_io_completion(cmd: *mut scsi_cmnd, good_bytes: u32) {
let mut result: i32 = unsafe { (*cmd).result };
let req: *mut request = scsi_cmd_to_rq(cmd);
let mut blk_stat: u8 = 0;
if unlikely__2!(result) != 0 {
result = scsi_io_completion_nz_result(cmd, result, unsafe { &mut (*core::ptr::addr_of_mut!(blk_stat)) });
}
// a nz result may or may not be an error
// Next deal with any sectors which we were able to correctly
// handle.
SCSI_LOG_HLCOMPLETE!();
// Failed, zero length commands always need to drop down
// to retry code. Fast path should return in this block.
if likely!(blk_rq_bytes(req as *const request) > 0 || (blk_stat as i32) == 0) != 0 {
if likely!(!scsi_end_request(req, blk_stat, good_bytes)) != 0 {
return;
}
// no bytes remaining
}
// Kill remainder if no retries.
let __v1: bool = blk_stat != 0 && unsafe { scsi_noretry_cmd(cmd) };
if __v1 {
if scsi_end_request(req, blk_stat, blk_rq_bytes(req as *const request)) {
let mut __v2: i32;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"Bytes remaining after failed, no-retry command".as_ptr() as *const u8) };
let mut __v3: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v3 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v4: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_scsi_io_completion_5, sym __slate_asm_string_scsi_io_completion_6, const 1120 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v4 = bug;
unsafe { __v3.unwrap()(__v4) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v2 = __ret_warn_on;
}
return;
}
// If there had been no error, but we have leftover bytes in the
// request just queue the command up again.
if likely!(result == 0) != 0 {
scsi_mq_requeue_cmd(cmd, 0u64);
} else {
scsi_io_completion_action(cmd, result);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_659: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

fn scsi_cmd_needs_dma_drain(sdev: *mut scsi_device, rq: *mut request) -> bool {
let __v0: bool = (unsafe { (*sdev).dma_drain_len }) != 0 && blk_rq_is_passthrough(rq);
let __v1: bool = __v0 && !op_is_write(req_op(rq as *const request) as u32);
__v1 && unsafe { unsafe { (*(*(*sdev).host).hostt).dma_need_drain }.unwrap()(rq) }
}

/// scsi_alloc_sgtables - Allocate and initialize data and integrity scatterlists
/// @cmd: SCSI command data structure to initialize.
///
/// Initializes @cmd->sdb and also @cmd->prot_sdb if data integrity is enabled
/// for @cmd.
///
/// Returns:
/// * BLK_STS_OK       - on success
/// * BLK_STS_RESOURCE - if the failure is retryable
/// * BLK_STS_IOERR    - if the failure is fatal
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_alloc_sgtables")]
extern "C" fn scsi_alloc_sgtables(cmd: *mut scsi_cmnd) -> u8 {
let mut __ret_warn_on: i32 = unsafe { core::mem::zeroed() };
let mut prot_sdb: *mut scsi_data_buffer = unsafe { core::mem::zeroed() };
let mut __ret_warn_on_1: i32 = unsafe { core::mem::zeroed() };
let mut count: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_22: *mut i32 = core::ptr::addr_of_mut!(count) as *mut i32;
let mut need_drain: bool = unsafe { core::mem::zeroed() };
let mut ret: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_24: *mut u8 = core::ptr::addr_of_mut!(ret) as *mut u8;
let mut last_sg: *mut scatterlist = unsafe { core::mem::zeroed() };
let mut nr_segs: u16 = unsafe { core::mem::zeroed() };
let mut rq: *mut request = unsafe { core::mem::zeroed() };
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
unsafe {
sdev = (*cmd).device;
rq = scsi_cmd_to_rq(cmd);
nr_segs = blk_rq_nr_phys_segments(rq);
last_sg = core::ptr::null_mut::<scatterlist>();
let __slate_auto_bytes = __slate_slot_24 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
need_drain = scsi_cmd_needs_dma_drain(sdev, rq);
let __slate_auto_bytes = __slate_slot_22 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
__ret_warn_on_1 = (nr_segs == 0) as i32;
if unlikely__2!(__ret_warn_on_1) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("660: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 660b - ., 3; .popsection\n# {0}", const 660 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_alloc_sgtables_33, sym __slate_asm_string_scsi_alloc_sgtables_34, const 1164 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("661: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 661b - ., 4; .popsection\n# {0}", const 661 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v21: i64 = (__ret_warn_on_1 != 0) as i64;
if __v21 != 0 {
return 10;
} else {
// Make sure there is space for the drain.  The driver must adjust
// max_hw_segments to be prepared for this.
if need_drain {
nr_segs = (nr_segs as i32).wrapping_add(1) as u16;
}
// If sg table allocation fails, requeue request later.
if unlikely__2!(sg_alloc_table_chained(core::ptr::addr_of_mut!((*cmd).sdb.table), nr_segs as i32, (*cmd).sdb.table.sgl, 2u32)) != 0 {
return 9;
} else {
// Next, walk the list, and fill in the addresses and sizes of
// each segment.
*__slate_slot_22 = __blk_rq_map_sg(rq, (*cmd).sdb.table.sgl, core::ptr::addr_of_mut!(last_sg));
if blk_rq_bytes(rq as *const request) & (*(*rq).q).limits.dma_pad_mask != 0 {
let pad_len: u32 = ((*(*rq).q).limits.dma_pad_mask & !blk_rq_bytes(rq as *const request)).wrapping_add(1);
let data_len: u32 = (*last_sg).length;
let __v13: u32 = (*last_sg).length;
(*last_sg).length = __v13.wrapping_add(pad_len);
sg_zero_buffer(last_sg, 1u32, pad_len as u64, data_len as i64);
let __v10: u32 = (*cmd).extra_len;
(*cmd).extra_len = __v10.wrapping_add(pad_len);
}
if need_drain {
sg_unmark_end(last_sg);
last_sg = sg_next(last_sg);
sg_set_buf(last_sg, (*sdev).dma_drain_buf as *const (), (*sdev).dma_drain_len as u32);
sg_mark_end(last_sg);
(*cmd).extra_len = ((*cmd).extra_len as u64).wrapping_add((*sdev).dma_drain_len) as u32;
let __v5: i32 = *__slate_slot_22;
*__slate_slot_22 = __v5.wrapping_add(1);
}
'__join_14: {
loop {
if unlikely!((*__slate_slot_22 as u32) > (*cmd).sdb.table.nents) != 0 {
break '__join_14;
} else {
if true {
break;
}
}
}
'__join_1: {
(*cmd).sdb.table.nents = *__slate_slot_22 as u32;
(*cmd).sdb.length = blk_rq_payload_bytes(rq);
if blk_integrity_rq(rq as *const request) {
prot_sdb = (*cmd).prot_sdb;
__ret_warn_on = prot_sdb.is_null() as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("663: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 663b - ., 3; .popsection\n# {0}", const 663 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_alloc_sgtables_29, sym __slate_asm_string_scsi_alloc_sgtables_30, const 1214 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("664: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 664b - ., 4; .popsection\n# {0}", const 664 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
let __v2: i64 = (__ret_warn_on != 0) as i64;
if __v2 != 0 {
// This can happen if someone (e.g. multipath)
// queues a command to a device on an adapter
// that does not support DIX.
*__slate_slot_24 = 10;
} else {
if sg_alloc_table_chained(core::ptr::addr_of_mut!((*prot_sdb).table), (*rq).nr_integrity_segments as i32, (*prot_sdb).table.sgl, 1u32) != 0 {
*__slate_slot_24 = 9;
} else {
*__slate_slot_22 = blk_rq_map_integrity_sg(rq, (*prot_sdb).table.sgl);
(*cmd).prot_sdb = prot_sdb;
(*(*cmd).prot_sdb).table.nents = *__slate_slot_22 as u32;
break '__join_1;
}
}
scsi_free_sgtables(cmd);
return *__slate_slot_24;
}
}
return 0;
}
core::arch::asm!("662: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 662b - ., 3; .popsection\n# {0}", const 662 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_scsi_alloc_sgtables_31, sym __slate_asm_string_scsi_alloc_sgtables_32, const 1207 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}
unsafe { core::mem::zeroed() }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_alloc_sgtables_665: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_cmnd) -> u8>, *mut ()>(Some(scsi_alloc_sgtables)) };

/// scsi_initialize_rq - initialize struct scsi_cmnd partially
/// @rq: Request associated with the SCSI command to be initialized.
///
/// This function initializes the members of struct scsi_cmnd that must be
/// initialized before request processing starts and that won't be
/// reinitialized if a SCSI command is requeued.
fn scsi_initialize_rq(rq: *mut request) {
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
unsafe { memset(((*cmd).cmnd.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 32]>() as u64) };
unsafe {
(*cmd).cmd_len = 16;
}
unsafe {
(*cmd).sense_len = 0;
}
init_rcu_head(unsafe { core::ptr::addr_of_mut!((*cmd).rcu) });
unsafe {
(*cmd).jiffies_at_alloc = core::ptr::read_volatile(core::ptr::addr_of!(jiffies));
}
unsafe {
(*cmd).retries = 0;
}
}

/// scsi_alloc_request - allocate a block request and partially
///                      initialize its &scsi_cmnd
/// @q: the device's request queue
/// @opf: the request operation code
/// @flags: block layer allocation flags
///
/// Return: &struct request pointer on success or %NULL on failure
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_alloc_request")]
extern "C" fn scsi_alloc_request(q: *mut request_queue, opf: u32, flags: u32) -> *mut request {
let rq: *mut request = unsafe { blk_mq_alloc_request(q, opf, flags) };
if !IS_ERR(rq as *const ()) {
scsi_initialize_rq(rq);
}
rq
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_alloc_request_666: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut request_queue, u32, u32) -> *mut request>, *mut ()>(Some(scsi_alloc_request)) };

/// Only called when the request isn't completed by SCSI, and not freed by
/// SCSI
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_cleanup_rq")]
extern "C" fn scsi_cleanup_rq(rq: *mut request) {
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
unsafe {
(*cmd).flags = 0;
}
if (unsafe { (*rq).rq_flags }) & 8u32 != 0 {
scsi_mq_uninit_cmd(cmd);
let __v2: u32 = (unsafe { (*rq).rq_flags }) & 4294967287u32;
unsafe {
(*rq).rq_flags = __v2;
}
}
}

/// Called before a request is prepared. See also scsi_mq_prep_fn().
#[unsafe(no_mangle)]
extern "C" fn scsi_init_command(dev: *mut scsi_device, cmd: *mut scsi_cmnd) {
let mut __slate_compound_3: __SlateRecord_types_h_197 = unsafe { core::mem::zeroed() };
let rq: *mut request = scsi_cmd_to_rq(cmd);
if !blk_rq_is_passthrough(rq) && (unsafe { (*cmd).flags }) & 2i32 == 0 {
unsafe {
(*cmd).flags = (*cmd).flags | 2i32;
}
scsi_initialize_rq(rq);
}
unsafe {
(*cmd).device = dev;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*cmd).eh_entry) });
'__slate_break_6: loop {
'__slate_break_5: loop {
'__slate_break_4: loop {
__init_work(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*cmd).abort_work)).work) }, 0i32);
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*cmd).abort_work)).work)).data = *{
__slate_compound_3 = __SlateRecord_types_h_197 { counter: 4503599625273344i64 };
core::ptr::addr_of_mut!(__slate_compound_3)
};
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*cmd).abort_work)).work)).entry) });
unsafe {
(*core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*cmd).abort_work)).work)).func = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut work_struct)>>(scmd_eh_abort_handler as *const ()) };
}
if true {
break;
}
}
if true {
break;
}
}
__timer_init!(unsafe { core::ptr::addr_of_mut!((*core::ptr::addr_of_mut!((*cmd).abort_work)).timer) }, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut timer_list)>>(delayed_work_timer_fn as *const ()) }, 2097152i32);
if true {
break;
}
}
}

static mut __key: lock_class_key = unsafe { core::mem::zeroed() };

fn scsi_setup_scsi_cmnd(sdev: *mut scsi_device, req: *mut request) -> u8 {
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(req) as *mut scsi_cmnd;
// Passthrough requests may transfer data, in which case they must
// a bio attached to them.  Or they might contain a SCSI command
// that does not transfer data, in which case they may optionally
// submit a request without an attached bio.
if !unsafe { (*req).bio }.is_null() {
let ret: u8 = scsi_alloc_sgtables(cmd);
if unlikely!((ret as i32) != 0) != 0 {
return ret;
}
} else {
loop {
if unlikely__3!(blk_rq_bytes(req as *const request)) != 0 {
loop {
unsafe {
core::arch::asm!("667: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 667b - ., 3; .popsection\n# {0}", const 667 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_scsi_setup_scsi_cmnd_0, sym __slate_asm_string_scsi_setup_scsi_cmnd_1, const 1332 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
unsafe { memset(core::ptr::addr_of_mut!((*cmd).sdb) as *mut (), 0i32, core::mem::size_of::<scsi_data_buffer>() as u64) };
}
unsafe {
(*cmd).transfersize = blk_rq_bytes(req as *const request);
}
0
}

fn scsi_device_state_check(sdev: *mut scsi_device, req: *mut request) -> u8 {
match (unsafe { (*sdev).sdev_state }) as u32 {
1 => {
return 0;
}
6 | 7 => {
if !unsafe { (*sdev).offline_already } {
unsafe {
(*sdev).offline_already = true;
}
unsafe { sdev_prefix_printk(c"\x013".as_ptr() as *const u8, sdev as *const scsi_device, core::ptr::null::<u8>(), c"rejecting I/O to offline device\n".as_ptr() as *const u8) };
}
// If the device is offline we refuse to process any
// commands.  The device must be brought online
// before trying any recovery commands.
return 10;
}
4 => {
unsafe { sdev_prefix_printk(c"\x013".as_ptr() as *const u8, sdev as *const scsi_device, core::ptr::null::<u8>(), c"rejecting I/O to dead device\n".as_ptr() as *const u8) };
// If the device is fully deleted, we refuse to
// process any commands as well.
return 10;
}
8 | 9 => {
return 9;
}
5 => {
let __v0: bool = if !req.is_null() {
let mut __v1: i64;
let __ret_warn_on: i32 = ((unsafe { (*req).rq_flags }) & 512u32 == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("668: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 668b - ., 3; .popsection\n# {0}", const 668 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_device_state_check_2, sym __slate_asm_string_scsi_device_state_check_3, const 1376 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("669: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 669b - ., 4; .popsection\n# {0}", const 669 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v1 = (__ret_warn_on != 0) as i64;
__v1 != 0
} else {
false
};
if __v0 {
return 9;
}
// If the device is blocked we only accept power management
// commands.
return 0;
}
_ => {
if !req.is_null() && (unsafe { (*req).rq_flags }) & 512u32 == 0 {
return 16;
}
// For any other not fully online state we only allow
// power management commands.
return 0;
}
}
unsafe { core::mem::zeroed() }
}

/// scsi_dev_queue_ready: if we can send requests to sdev, assign one token
/// and return the token else return -1.
fn scsi_dev_queue_ready(q: *mut request_queue, sdev: *mut scsi_device) -> i32 {
let mut token: i32 = 0;
if unsafe { (*sdev).budget_map.map }.is_null() {
return 2147483647i32;
}
token = unsafe { sbitmap_get(core::ptr::addr_of_mut!((*sdev).budget_map)) };
if token < 0 {
return 1i32.wrapping_neg();
}
if atomic_read((unsafe { core::ptr::addr_of_mut!((*sdev).device_blocked) }) as *const __SlateRecord_types_h_190) == 0 {
return token;
}
// Only unblock if no other commands are pending and
// if device_blocked has decreased to zero
let __v0: bool = scsi_device_busy(sdev) > 1 || atomic_dec_return(unsafe { core::ptr::addr_of_mut!((*sdev).device_blocked) }) > 0;
if __v0 {
sbitmap_put(unsafe { core::ptr::addr_of_mut!((*sdev).budget_map) }, token as u32);
return 1i32.wrapping_neg();
}
SCSI_LOG_MLQUEUE!();
token
}

/// scsi_target_queue_ready: checks if there we can send commands to target
/// @sdev: scsi device on starget to check.
fn scsi_target_queue_ready(shost: *mut Scsi_Host, sdev: *mut scsi_device) -> i32 {
let mut busy: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut u32 = core::ptr::addr_of_mut!(busy) as *mut u32;
let mut starget: *mut scsi_target = unsafe { core::mem::zeroed() };
unsafe {
starget = scsi_target(sdev);
let __slate_auto_bytes = __slate_slot_0 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
if ((*starget).__slate_bits_0.__get_single_lun() as i32) != 0 {
spin_lock_irq((*shost).host_lock);
if !unsafe { (*starget).starget_sdev_user }.is_null() && (*starget).starget_sdev_user != sdev {
spin_unlock_irq((*shost).host_lock);
return 0;
} else {
(*starget).starget_sdev_user = sdev;
spin_unlock_irq((*shost).host_lock);
}
}
if (*starget).can_queue <= 0 {
return 1;
} else {
'__join_2: {
'__join_3: {
*__slate_slot_0 = atomic_inc_return(core::ptr::addr_of_mut!((*starget).target_busy)).wrapping_sub(1) as u32;
if atomic_read(core::ptr::addr_of_mut!((*starget).target_blocked) as *const __SlateRecord_types_h_190) > 0 {
if *__slate_slot_0 != 0 {
break '__join_3;
} else {
// unblock after target_blocked iterates to zero
if atomic_dec_return(core::ptr::addr_of_mut!((*starget).target_blocked)) > 0 {
break '__join_2;
} else {
SCSI_LOG_MLQUEUE!();
}
}
}
if !(*__slate_slot_0 >= (*starget).can_queue) {
return 1;
}
}
spin_lock_irq((*shost).host_lock);
list_move_tail(core::ptr::addr_of_mut!((*sdev).starved_entry), core::ptr::addr_of_mut!((*shost).starved_list));
spin_unlock_irq((*shost).host_lock);
}
if (*starget).can_queue > 0 {
atomic_dec(core::ptr::addr_of_mut!((*starget).target_busy));
}
return 0;
}
}
}

/// scsi_host_queue_ready: if we can send requests to shost, return 1 else
/// return 0. We must end up running the queue again whenever 0 is
/// returned, else IO can hang.
fn scsi_host_queue_ready(q: *mut request_queue, shost: *mut Scsi_Host, sdev: *mut scsi_device, cmd: *mut scsi_cmnd) -> i32 {
unsafe {
'__join_0: {
'__join_3: {
if atomic_read(core::ptr::addr_of_mut!((*shost).host_blocked) as *const __SlateRecord_types_h_190) > 0 {
if scsi_host_busy(shost) > 0 {
break '__join_3;
} else {
// unblock after host_blocked iterates to zero
if atomic_dec_return(core::ptr::addr_of_mut!((*shost).host_blocked)) > 0 {
break '__join_0;
} else {
SCSI_LOG_MLQUEUE!();
}
}
}
if ((*shost).__slate_bits_0.__get_host_self_blocked() as i32) == 0 {
// We're OK to process the command, so we can't be starved
if list_empty(core::ptr::addr_of_mut!((*sdev).starved_entry) as *const list_head) == 0 {
spin_lock_irq((*shost).host_lock);
if list_empty(core::ptr::addr_of_mut!((*sdev).starved_entry) as *const list_head) == 0 {
list_del_init(core::ptr::addr_of_mut!((*sdev).starved_entry));
}
spin_unlock_irq((*shost).host_lock);
}
__set_bit!(1i32, core::ptr::addr_of_mut!((*cmd).state));
return 1;
}
}
spin_lock_irq((*shost).host_lock);
if list_empty(core::ptr::addr_of_mut!((*sdev).starved_entry) as *const list_head) != 0 {
list_add_tail(core::ptr::addr_of_mut!((*sdev).starved_entry), core::ptr::addr_of_mut!((*shost).starved_list));
}
spin_unlock_irq((*shost).host_lock);
}
scsi_dec_host_busy(shost, cmd);
return 0;
}
}

/// Busy state exporting function for request stacking drivers.
///
/// For efficiency, no lock is taken to check the busy state of
/// shost/starget/sdev, since the returned value is not guaranteed and
/// may be changed after request stacking drivers call the function,
/// regardless of taking lock or not.
///
/// When scsi can't dispatch I/Os anymore and needs to kill I/Os scsi
/// needs to return 'not busy'. Otherwise, request stacking drivers
/// may hold requests forever.
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_lld_busy")]
extern "C" fn scsi_mq_lld_busy(q: *mut request_queue) -> bool {
let sdev: *mut scsi_device = (unsafe { (*q).queuedata }) as *mut scsi_device;
if blk_queue_dying!(q) != 0 {
return false;
}
let shost: *mut Scsi_Host = unsafe { (*sdev).host };
// Ignore host/starget busy state.
// Since block layer does not have a concept of fairness across
// multiple queues, congestion of host/starget needs to be handled
// in SCSI layer.
let __v1: bool = scsi_host_in_recovery(shost) != 0 || scsi_device_is_busy(sdev);
if __v1 {
return true;
}
false
}

/// Block layer request completion callback. May be called from interrupt
/// context.
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_complete")]
extern "C" fn scsi_complete(rq: *mut request) {
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
let mut disposition: u32 = 0;
if blk_mq_is_reserved_rq(rq) {
// Only pass-through requests are supported in this code path.
let mut __v0: i64;
let __ret_warn_on: i32 = !(!(!blk_rq_is_passthrough(scsi_cmd_to_rq(cmd)))) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("670: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 670b - ., 3; .popsection\n# {0}", const 670 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_complete_1, sym __slate_asm_string_scsi_complete_2, const 1574 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("671: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 671b - ., 4; .popsection\n# {0}", const 671 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
scsi_mq_uninit_cmd(cmd);
unsafe { __blk_mq_end_request(rq, scsi_result_to_blk_status((*cmd).result)) };
return;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*cmd).eh_entry) });
atomic_inc(unsafe { core::ptr::addr_of_mut!((*(*cmd).device).iodone_cnt) });
if (unsafe { (*cmd).result }) != 0 {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*(*cmd).device).ioerr_cnt) });
}
disposition = unsafe { scsi_decide_disposition(cmd) };
let __v5: bool = (disposition as u32) != 8194 && scsi_cmd_runtime_exceeced(cmd);
if __v5 {
disposition = 8194;
}
scsi_log_completion(cmd, (disposition as u32) as i32);
match disposition as u32 {
8194 => {
unsafe { scsi_finish_command(cmd) };
}
8193 => {
scsi_queue_insert(cmd, 4183u32);
}
8198 => {
scsi_queue_insert(cmd, 4182u32);
}
_ => {
unsafe { scsi_eh_scmd_add(cmd) };
}
}
}

/// scsi_dispatch_cmd - Dispatch a command to the low-level driver.
/// @cmd: command block we are dispatching.
///
/// Return: nonzero return request was rejected and device's queue needs to be
/// plugged.
fn scsi_dispatch_cmd(cmd: *mut scsi_cmnd) -> u32 {
let mut rtn: i32 = unsafe { core::mem::zeroed() };
let mut host: *mut Scsi_Host = unsafe { core::mem::zeroed() };
unsafe {
host = (*(*cmd).device).host;
rtn = 0;
atomic_inc(core::ptr::addr_of_mut!((*(*cmd).device).iorequest_cnt));
// check if the device is still usable
if unlikely!(((*(*cmd).device).sdev_state as u32) == 4) != 0 {
// in SDEV_DEL we error all commands. DID_NO_CONNECT
// returns an immediate error upwards, and signals
// that the device is no longer present
(*cmd).result = 65536i32;
} else {
// Check to see if the scsi lld made this device blocked.
if unlikely__2!(scsi_device_blocked((*cmd).device)) != 0 {
// in blocked state, the command is just put back on
// the device queue.  The suspend state has already
// blocked the queue so future requests should not
// occur until the device transitions out of the
// suspend state.
SCSI_LOG_MLQUEUE!();
atomic_dec(core::ptr::addr_of_mut!((*(*cmd).device).iorequest_cnt));
return 4182;
} else {
// Store the LUN value in cmnd, if needed.
if ((*(*cmd).device).__slate_bits_0.__get_lun_in_cdb() as i32) != 0 {
*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) = ((((*((*cmd).cmnd.as_mut_ptr() as *mut u8).wrapping_offset(1 as isize) as i32) & 31) as u64) | (*(*cmd).device).lun << 5i32 & 224) as u8;
}
scsi_log_send(cmd);
// Before we queue this command, check if the command
// length exceeds what the host adapter can handle.
if ((*cmd).cmd_len as i32) > ((*(*(*cmd).device).host).max_cmd_len as i32) {
SCSI_LOG_MLQUEUE!();
(*cmd).result = 327680i32;
} else {
if unlikely!((scsi_get_host_state(host) as u32) == 4) != 0 {
(*cmd).result = 65536i32;
} else {
trace_scsi_dispatch_cmd_start(cmd);
rtn = (unsafe { (*(*host).hostt).queuecommand }.unwrap()(host, cmd) as u32) as i32;
if rtn != 0 {
atomic_dec(core::ptr::addr_of_mut!((*(*cmd).device).iorequest_cnt));
trace_scsi_dispatch_cmd_error(cmd, rtn);
if rtn != 4182 && rtn != 4184 {
rtn = 4181;
}
SCSI_LOG_MLQUEUE!();
}
return rtn as u32;
}
}
}
}
scsi_done(cmd);
return 0;
}
}

/// Size in bytes of the sg-list stored in the scsi-mq command-private data.
fn scsi_mq_inline_sgl_size(shost: *mut Scsi_Host) -> u32 {
(min_t__5!(u32, unsafe { (*shost).sg_tablesize }, 2i32) as u64).wrapping_mul(core::mem::size_of::<scatterlist>() as u64) as u32
}

fn scsi_prepare_cmd(req: *mut request) -> u8 {
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(req) as *mut scsi_cmnd;
let sdev: *mut scsi_device = (unsafe { (*(*req).q).queuedata }) as *mut scsi_device;
let shost: *mut Scsi_Host = unsafe { (*sdev).host };
let in_flight: bool = test_bit!(1i32, unsafe { core::ptr::addr_of_mut!((*cmd).state) }) != 0;
scsi_init_command(sdev, cmd);
unsafe {
(*cmd).eh_eflags = 0;
}
unsafe {
(*cmd).prot_type = 0;
}
unsafe {
(*cmd).prot_flags = 0;
}
unsafe {
(*cmd).submitter = 0;
}
unsafe { memset(core::ptr::addr_of_mut!((*cmd).sdb) as *mut (), 0i32, core::mem::size_of::<scsi_data_buffer>() as u64) };
unsafe {
(*cmd).underflow = 0;
}
unsafe {
(*cmd).transfersize = 0;
}
unsafe {
(*cmd).host_scribble = core::ptr::null_mut::<u8>();
}
unsafe {
(*cmd).result = 0;
}
unsafe {
(*cmd).extra_len = 0;
}
unsafe {
(*cmd).state = 0;
}
if in_flight {
__set_bit!(1i32, unsafe { core::ptr::addr_of_mut!((*cmd).state) });
}
unsafe {
(*cmd).prot_op = 0;
}
if blk_rq_bytes(req as *const request) != 0 {
unsafe {
(*cmd).sc_data_direction = rq_dma_dir!(req) as u32;
}
} else {
unsafe {
(*cmd).sc_data_direction = 3;
}
}
let sg: *mut scatterlist = ((unsafe { (((cmd as *const u8).wrapping_offset(core::mem::size_of::<scsi_cmnd>() as isize) as *mut ()) as *const u8).wrapping_offset((*(*shost).hostt).cmd_size as isize) }) as *mut ()) as *mut scatterlist;
unsafe {
(*cmd).sdb.table.sgl = sg;
}
if scsi_host_get_prot(shost) != 0 {
unsafe { memset((*cmd).prot_sdb as *mut (), 0i32, core::mem::size_of::<scsi_data_buffer>() as u64) };
unsafe {
(*(*cmd).prot_sdb).table.sgl = unsafe { (*cmd).prot_sdb }.wrapping_offset(1 as isize) as *mut scatterlist;
}
}
// Special handling for passthrough commands, which don't go to the ULP
// at all:
if blk_rq_is_passthrough(req) {
return scsi_setup_scsi_cmnd(sdev, req);
}
if !unsafe { (*sdev).handler }.is_null() && (unsafe { (*(*sdev).handler).prep_fn }) != None {
let ret: u8 = unsafe { unsafe { (*(*sdev).handler).prep_fn }.unwrap()(sdev, req) };
if (ret as i32) != 0 {
return ret;
}
}
// Usually overridden by the ULP
unsafe {
(*cmd).allowed = 0;
}
unsafe { memset(((*cmd).cmnd.as_mut_ptr() as *mut u8) as *mut (), 0i32, core::mem::size_of::<[u8; 32]>() as u64) };
unsafe { unsafe { (*scsi_cmd_to_driver(cmd)).init_command }.unwrap()(cmd) }
}

fn scsi_done_internal(cmd: *mut scsi_cmnd, complete_directly: bool) {
let req: *mut request = scsi_cmd_to_rq(cmd);
match ((unsafe { (*cmd).submitter }) as u8) as i32 {
0 => {
}
1 => {
unsafe { scsi_eh_done(cmd) };
return;
}
2 => {
return;
}
_ => {
}
}
if unlikely!(blk_should_fake_timeout(unsafe { (*scsi_cmd_to_rq(cmd)).q })) != 0 {
return;
}
if unlikely!(test_and_set_bit(0i64, (unsafe { core::ptr::addr_of_mut!((*cmd).state) }) as *mut u64)) != 0 {
return;
}
trace_scsi_dispatch_cmd_done(cmd);
if complete_directly {
blk_mq_complete_request_direct(req, Some(scsi_complete));
} else {
unsafe { blk_mq_complete_request(req) };
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_done")]
extern "C" fn scsi_done(cmd: *mut scsi_cmnd) {
scsi_done_internal(cmd, false);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_done_674: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_cmnd)>, *mut ()>(Some(scsi_done)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_done_direct")]
extern "C" fn scsi_done_direct(cmd: *mut scsi_cmnd) {
scsi_done_internal(cmd, true);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_done_direct_675: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_cmnd)>, *mut ()>(Some(scsi_done_direct)) };

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_put_budget")]
extern "C" fn scsi_mq_put_budget(q: *mut request_queue, budget_token: i32) {
let sdev: *mut scsi_device = (unsafe { (*q).queuedata }) as *mut scsi_device;
if !unsafe { (*sdev).budget_map.map }.is_null() {
sbitmap_put(unsafe { core::ptr::addr_of_mut!((*sdev).budget_map) }, budget_token as u32);
}
}

// When to reinvoke queueing after a resource shortage. It's 3 msecs to
// not change behaviour from the previous unplug mechanism, experimentation
// may prove this needs changing.
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_get_budget")]
extern "C" fn scsi_mq_get_budget(q: *mut request_queue) -> i32 {
let sdev: *mut scsi_device = (unsafe { (*q).queuedata }) as *mut scsi_device;
let token: i32 = scsi_dev_queue_ready(q, sdev);
if token >= 0 {
return token;
}
atomic_inc(unsafe { core::ptr::addr_of_mut!((*sdev).restarts) });
// Orders atomic_inc(&sdev->restarts) and atomic_read(&sdev->device_busy).
// .restarts must be incremented before .device_busy is read because the
// code in scsi_run_queue_async() depends on the order of these operations.
smp_mb__after_atomic!();
// If all in-flight requests originated from this LUN are completed
// before reading .device_busy, sdev->device_busy will be observed as
// zero, then blk_mq_delay_run_hw_queues() will dispatch this request
// soon. Otherwise, completion of one of these requests will observe
// the .restarts flag, and the request queue will be run for handling
// this request, see scsi_end_request().
let __v3: bool = scsi_device_busy(sdev) == 0 && scsi_device_blocked(sdev) == 0;
if __v3 {
unsafe { blk_mq_delay_run_hw_queues((*sdev).request_queue, 3u64) };
}
1i32.wrapping_neg()
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_set_rq_budget_token")]
extern "C" fn scsi_mq_set_rq_budget_token(req: *mut request, token: i32) {
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(req) as *mut scsi_cmnd;
unsafe {
(*cmd).budget_token = token;
}
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_get_rq_budget_token")]
extern "C" fn scsi_mq_get_rq_budget_token(req: *mut request) -> i32 {
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(req) as *mut scsi_cmnd;
unsafe { (*cmd).budget_token }
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_queue_rq")]
extern "C" fn scsi_queue_rq(hctx: *mut blk_mq_hw_ctx, bd: *const blk_mq_queue_data) -> u8 {
let mut reason: core::mem::MaybeUninit<u32> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut u32 = core::ptr::addr_of_mut!(reason) as *mut u32;
let mut ret: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut u8 = core::ptr::addr_of_mut!(ret) as *mut u8;
let mut cmd: *mut scsi_cmnd = unsafe { core::mem::zeroed() };
let mut shost: *mut Scsi_Host = unsafe { core::mem::zeroed() };
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut q: *mut request_queue = unsafe { core::mem::zeroed() };
let mut req: *mut request = unsafe { core::mem::zeroed() };
unsafe {
req = (*bd).rq;
q = (*req).q;
sdev = (*q).queuedata as *mut scsi_device;
shost = (*sdev).host;
cmd = blk_mq_rq_to_pdu(req) as *mut scsi_cmnd;
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_15 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __ret_warn_on: i32 = ((*cmd).budget_token < 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
core::arch::asm!("676: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 676b - ., 3; .popsection\n# {0}", const 676 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_queue_rq_22, sym __slate_asm_string_scsi_queue_rq_23, const 1865 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
break;
}
}
core::arch::asm!("677: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 677b - ., 4; .popsection\n# {0}", const 677 as i32, options(nostack, att_syntax));
if true {
break;
}
}
}
'__join_11: {
'__join_13: {
// Bypass the SCSI device, SCSI target and SCSI host checks for
// reserved commands.
if !blk_mq_is_reserved_rq(req) {
// If the device is not in running state we will reject some or
// all commands.
if unlikely!(((*sdev).sdev_state as u32) != 2) != 0 {
*__slate_slot_16 = scsi_device_state_check(sdev, req);
if (*__slate_slot_16 as i32) != 0 {
break '__join_11;
}
}
*__slate_slot_16 = 9;
if scsi_target_queue_ready(shost, sdev) == 0 {
break '__join_11;
} else {
if unlikely__2!(scsi_host_in_recovery(shost)) != 0 {
if (*cmd).flags & 16i32 != 0 {
*__slate_slot_16 = 16;
break '__join_13;
} else {
break '__join_13;
}
} else {
if scsi_host_queue_ready(q, shost, sdev, cmd) == 0 {
break '__join_13;
}
}
}
}
// Only clear the driver-private command data if the LLD does not supply
// a function to initialize that data.
if (*(*shost).hostt).cmd_size != 0 && (*(*shost).hostt).init_cmd_priv == None {
memset(scsi_cmd_priv(cmd), 0i32, (*(*shost).hostt).cmd_size as u64);
}
'__join_14: {
if (*req).rq_flags & 8u32 == 0 {
*__slate_slot_16 = scsi_prepare_cmd(req);
if (*__slate_slot_16 as i32) != 0 {
break '__join_14;
} else {
(*req).rq_flags = (*req).rq_flags | 8u32;
}
} else {
clear_bit(0i64, core::ptr::addr_of_mut!((*cmd).state) as *mut u64);
}
(*cmd).flags = (*cmd).flags & 18i32;
if ((*sdev).__slate_bits_0.__get_simple_tags() as i32) != 0 {
(*cmd).flags = (*cmd).flags | 1i32;
}
if (*bd).last {
(*cmd).flags = (*cmd).flags | 4i32;
}
scsi_set_resid(cmd, 0u32);
memset((*cmd).sense_buffer as *mut (), 0i32, 96u64);
(*cmd).submitter = 0;
blk_mq_start_request(req);
if blk_mq_is_reserved_rq(req) {
*__slate_slot_15 = unsafe { (*(*shost).hostt).queue_reserved_command }.unwrap()(shost, cmd);
if (*__slate_slot_15 as u32) != 0 {
*__slate_slot_16 = 9;
break '__join_11;
} else {
return 0;
}
} else {
*__slate_slot_15 = scsi_dispatch_cmd(cmd);
if (*__slate_slot_15 as u32) != 0 {
scsi_set_blocked(cmd, *__slate_slot_15);
*__slate_slot_16 = 9;
} else {
return 0;
}
}
}
scsi_dec_host_busy(shost, cmd);
}
if (*scsi_target(sdev)).can_queue > 0 {
atomic_dec(core::ptr::addr_of_mut!((*scsi_target(sdev)).target_busy));
}
}
scsi_mq_put_budget(q, (*cmd).budget_token);
(*cmd).budget_token = 1i32.wrapping_neg();
let __t0: i32 = *__slate_slot_16 as i32;
if __t0 != 0 {
if __t0 == 9 {
if scsi_device_blocked(sdev) != 0 {
*__slate_slot_16 = 13;
}
} else {
if __t0 == 12 {
(*cmd).result = 131072i32;
if (*req).rq_flags & 8u32 != 0 {
scsi_mq_uninit_cmd(cmd);
}
} else {
if unlikely!(scsi_device_online(sdev) == 0) != 0 {
(*cmd).result = 65536i32;
} else {
(*cmd).result = 458752i32;
}
// Make sure to release all allocated resources when
// we hit an error, as we will never see this command
// again.
if (*req).rq_flags & 8u32 != 0 {
scsi_mq_uninit_cmd(cmd);
}
scsi_run_queue_async(sdev);
}
}
}
return *__slate_slot_16;
}
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_init_request")]
extern "C" fn scsi_mq_init_request(set: *mut blk_mq_tag_set, rq: *mut request, hctx_idx: u32, numa_node__0: i32) -> i32 {
let shost: *mut Scsi_Host = (unsafe { (*set).driver_data }) as *mut Scsi_Host;
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
let mut ret: i32 = 0;
unsafe {
(*cmd).sense_buffer = alloc_hooks!(kmem_cache_alloc_node_noprof(scsi_sense_cache, 3264u32, numa_node__0)) as *mut u8;
}
if unsafe { (*cmd).sense_buffer }.is_null() {
return 12i32.wrapping_neg();
}
if scsi_host_get_prot(shost) != 0 {
let sg: *mut scatterlist = ((unsafe { (((cmd as *const u8).wrapping_offset(core::mem::size_of::<scsi_cmnd>() as isize) as *mut ()) as *const u8).wrapping_offset((*(*shost).hostt).cmd_size as isize) }) as *mut ()) as *mut scatterlist;
unsafe {
(*cmd).prot_sdb = ((sg as *const u8).wrapping_offset(scsi_mq_inline_sgl_size(shost) as isize) as *mut ()) as *mut scsi_data_buffer;
}
}
if (unsafe { (*(*shost).hostt).init_cmd_priv }) != None {
ret = unsafe { unsafe { (*(*shost).hostt).init_cmd_priv }.unwrap()(shost, cmd) };
if ret < 0 {
unsafe { kmem_cache_free(scsi_sense_cache, (*cmd).sense_buffer as *mut ()) };
}
}
ret
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_exit_request")]
extern "C" fn scsi_mq_exit_request(set: *mut blk_mq_tag_set, rq: *mut request, hctx_idx: u32) {
let shost: *mut Scsi_Host = (unsafe { (*set).driver_data }) as *mut Scsi_Host;
let cmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
if (unsafe { (*(*shost).hostt).exit_cmd_priv }) != None {
unsafe { unsafe { (*(*shost).hostt).exit_cmd_priv }.unwrap()(shost, cmd) };
}
unsafe { kmem_cache_free(scsi_sense_cache, (*cmd).sense_buffer as *mut ()) };
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_poll")]
extern "C" fn scsi_mq_poll(hctx: *mut blk_mq_hw_ctx, iob: *mut io_comp_batch) -> i32 {
let shost: *mut Scsi_Host = (unsafe { (*hctx).driver_data }) as *mut Scsi_Host;
if (unsafe { (*(*shost).hostt).mq_poll }) != None {
return unsafe { unsafe { (*(*shost).hostt).mq_poll }.unwrap()(shost, (*hctx).queue_num) };
}
0
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_init_hctx")]
extern "C" fn scsi_init_hctx(hctx: *mut blk_mq_hw_ctx, data: *mut (), hctx_idx: u32) -> i32 {
let shost: *mut Scsi_Host = data as *mut Scsi_Host;
unsafe {
(*hctx).driver_data = shost as *mut ();
}
0
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_map_queues")]
extern "C" fn scsi_map_queues(set: *mut blk_mq_tag_set) {
let shost: *mut Scsi_Host = container_of__9!(set, Scsi_Host);
if (unsafe { (*(*shost).hostt).map_queues }) != None {
unsafe { unsafe { (*(*shost).hostt).map_queues }.unwrap()(shost) };
return;
}
unsafe { blk_mq_map_queues(((*set).map.as_mut_ptr() as *mut blk_mq_queue_map).wrapping_offset(0 as isize)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_init_limits_686: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut Scsi_Host, *mut queue_limits)>, *mut ()>(Some(scsi_init_limits)) };

static scsi_mq_ops_no_commit: __SlateConst<blk_mq_ops> = __SlateConst({
let mut __t0: blk_mq_ops = unsafe { core::mem::zeroed() };
__t0.queue_rq = Some(scsi_queue_rq);
__t0.get_budget = Some(scsi_mq_get_budget);
__t0.put_budget = Some(scsi_mq_put_budget);
__t0.set_rq_budget_token = Some(scsi_mq_set_rq_budget_token);
__t0.get_rq_budget_token = Some(scsi_mq_get_rq_budget_token);
__t0.timeout = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut request) -> u32>>(scsi_timeout as *const ()) };
__t0.poll = Some(scsi_mq_poll);
__t0.complete = Some(scsi_complete);
__t0.init_hctx = Some(scsi_init_hctx);
__t0.init_request = Some(scsi_mq_init_request);
__t0.exit_request = Some(scsi_mq_exit_request);
__t0.cleanup_rq = Some(scsi_cleanup_rq);
__t0.busy = Some(scsi_mq_lld_busy);
__t0.map_queues = Some(scsi_map_queues);
__t0.show_rq = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut seq_file, *mut request)>>(scsi_show_rq as *const ()) };
__t0
});

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_commit_rqs")]
extern "C" fn scsi_commit_rqs(hctx: *mut blk_mq_hw_ctx) {
let shost: *mut Scsi_Host = (unsafe { (*hctx).driver_data }) as *mut Scsi_Host;
unsafe { unsafe { (*(*shost).hostt).commit_rqs }.unwrap()(shost, (*hctx).queue_num as u16) };
}

static scsi_mq_ops: __SlateConst<blk_mq_ops> = __SlateConst({
let mut __t0: blk_mq_ops = unsafe { core::mem::zeroed() };
__t0.queue_rq = Some(scsi_queue_rq);
__t0.commit_rqs = Some(scsi_commit_rqs);
__t0.get_budget = Some(scsi_mq_get_budget);
__t0.put_budget = Some(scsi_mq_put_budget);
__t0.set_rq_budget_token = Some(scsi_mq_set_rq_budget_token);
__t0.get_rq_budget_token = Some(scsi_mq_get_rq_budget_token);
__t0.timeout = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut request) -> u32>>(scsi_timeout as *const ()) };
__t0.poll = Some(scsi_mq_poll);
__t0.complete = Some(scsi_complete);
__t0.init_hctx = Some(scsi_init_hctx);
__t0.init_request = Some(scsi_mq_init_request);
__t0.exit_request = Some(scsi_mq_exit_request);
__t0.cleanup_rq = Some(scsi_cleanup_rq);
__t0.busy = Some(scsi_mq_lld_busy);
__t0.map_queues = Some(scsi_map_queues);
__t0.show_rq = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut seq_file, *mut request)>>(scsi_show_rq as *const ()) };
__t0
});

#[unsafe(no_mangle)]
extern "C" fn scsi_mq_setup_tags(shost: *mut Scsi_Host) -> i32 {
let mut cmd_size: u32 = 0;
let mut sgl_size: u32 = 0;
let tag_set: *mut blk_mq_tag_set = unsafe { core::ptr::addr_of_mut!((*shost).tag_set) };
sgl_size = max_t!(u32, core::mem::size_of::<scatterlist>() as u64, scsi_mq_inline_sgl_size(shost));
cmd_size = (core::mem::size_of::<scsi_cmnd>() as u64).wrapping_add((unsafe { (*(*shost).hostt).cmd_size }) as u64).wrapping_add(sgl_size as u64) as u32;
if scsi_host_get_prot(shost) != 0 {
cmd_size = (cmd_size as u64).wrapping_add((core::mem::size_of::<scsi_data_buffer>() as u64).wrapping_add((core::mem::size_of::<scatterlist>() as u64).wrapping_mul(1))) as u32;
}
unsafe { memset(tag_set as *mut (), 0i32, core::mem::size_of::<blk_mq_tag_set>() as u64) };
if (unsafe { (*(*shost).hostt).commit_rqs }) != None {
unsafe {
(*tag_set).ops = core::ptr::addr_of!(scsi_mq_ops.0);
}
} else {
unsafe {
(*tag_set).ops = core::ptr::addr_of!(scsi_mq_ops_no_commit.0);
}
}
let __v5: u32 = unsafe { (*shost).nr_hw_queues };
unsafe {
(*tag_set).nr_hw_queues = if __v5 != 0 { __v5 } else { 1 };
}
let __v6: u32 = unsafe { (*shost).nr_maps };
unsafe {
(*tag_set).nr_maps = if __v6 != 0 { __v6 } else { 1 };
}
unsafe {
(*tag_set).queue_depth = ((*shost).can_queue as u32).wrapping_add((*shost).nr_reserved_cmds);
}
unsafe {
(*tag_set).reserved_tags = (*shost).nr_reserved_cmds;
}
unsafe {
(*tag_set).cmd_size = cmd_size;
}
unsafe {
(*tag_set).numa_node = dev_to_node((*shost).dma_dev);
}
if ((unsafe { (*(*shost).hostt).__slate_bits_0.__get_tag_alloc_policy_rr() }) as i32) != 0 {
unsafe {
(*tag_set).flags = (*tag_set).flags | 32;
}
}
if ((unsafe { (*shost).__slate_bits_0.__get_queuecommand_may_block() }) as i32) != 0 {
unsafe {
(*tag_set).flags = (*tag_set).flags | 16;
}
}
unsafe {
(*tag_set).driver_data = shost as *mut ();
}
if ((unsafe { (*shost).__slate_bits_0.__get_host_tagset() }) as i32) != 0 {
unsafe {
(*tag_set).flags = (*tag_set).flags | 8;
}
}
unsafe { blk_mq_alloc_tag_set(tag_set) }
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_mq_free_tags")]
extern "C" fn scsi_mq_free_tags(kref: *mut kref) {
let shost: *mut Scsi_Host = container_of__10!(kref);
unsafe { blk_mq_free_tag_set(core::ptr::addr_of_mut!((*shost).tag_set)) };
unsafe { complete(core::ptr::addr_of_mut!((*shost).tagset_freed)) };
}

/// scsi_get_internal_cmd() - Allocate an internal SCSI command.
/// @sdev: SCSI device from which to allocate the command
/// @data_direction: Data direction for the allocated command
/// @flags: request allocation flags, e.g. BLK_MQ_REQ_RESERVED or
/// 	BLK_MQ_REQ_NOWAIT.
///
/// Allocates a SCSI command for internal LLDD use.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_get_internal_cmd")]
extern "C" fn scsi_get_internal_cmd(sdev: *mut scsi_device, data_direction: u32, flags: u32) -> *mut scsi_cmnd {
let op: u32 = (if (data_direction as u32) == 1 { 35i32 } else { 34i32 }) as u32;
let rq: *mut request = scsi_alloc_request(unsafe { (*sdev).request_queue }, op as u32, flags);
if IS_ERR(rq as *const ()) {
return core::ptr::null_mut::<scsi_cmnd>();
}
let scmd: *mut scsi_cmnd = blk_mq_rq_to_pdu(rq) as *mut scsi_cmnd;
unsafe {
(*scmd).device = sdev;
}
scmd
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_get_internal_cmd_689: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, u32, u32) -> *mut scsi_cmnd>, *mut ()>(Some(scsi_get_internal_cmd)) };

/// scsi_put_internal_cmd() - Free an internal SCSI command.
/// @scmd: SCSI command to be freed
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_put_internal_cmd")]
extern "C" fn scsi_put_internal_cmd(scmd: *mut scsi_cmnd) {
unsafe { blk_mq_free_request(blk_mq_rq_from_pdu(scmd as *mut ())) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_put_internal_cmd_690: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_cmnd)>, *mut ()>(Some(scsi_put_internal_cmd)) };

/// scsi_device_from_queue - return sdev associated with a request_queue
/// @q: The request queue to return the sdev from
///
/// Return the sdev associated with a request queue or NULL if the
/// request_queue does not reference a SCSI device.
#[unsafe(no_mangle)]
extern "C" fn scsi_device_from_queue(q: *mut request_queue) -> *mut scsi_device {
let mut sdev: *mut scsi_device = core::ptr::null_mut::<scsi_device>();
if (unsafe { (*q).mq_ops }) == unsafe { core::ptr::addr_of!(scsi_mq_ops_no_commit.0) } || (unsafe { (*q).mq_ops }) == unsafe { core::ptr::addr_of!(scsi_mq_ops.0) } {
sdev = (unsafe { (*q).queuedata }) as *mut scsi_device;
}
let __v0: bool = sdev.is_null() || unsafe { get_device(core::ptr::addr_of_mut!((*sdev).sdev_gendev)) }.is_null();
if __v0 {
sdev = core::ptr::null_mut::<scsi_device>();
}
sdev
}

/// scsi_block_requests - Utility function used by low-level drivers to prevent
/// further commands from being queued to the device.
/// @shost:  host in question
///
/// There is no timer nor any other means by which the requests get unblocked
/// other than the low-level driver calling scsi_unblock_requests().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_block_requests")]
extern "C" fn scsi_block_requests(shost: *mut Scsi_Host) {
unsafe {
(*shost).__slate_bits_0.__set_host_self_blocked(1u32);
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_block_requests_691: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut Scsi_Host)>, *mut ()>(Some(scsi_block_requests)) };

/// scsi_unblock_requests - Utility function used by low-level drivers to allow
/// further commands to be queued to the device.
/// @shost:  host in question
///
/// There is no timer nor any other means by which the requests get unblocked
/// other than the low-level driver calling scsi_unblock_requests(). This is done
/// as an API function so that changes to the internals of the scsi mid-layer
/// won't require wholesale changes to drivers that use this feature.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_unblock_requests")]
extern "C" fn scsi_unblock_requests(shost: *mut Scsi_Host) {
unsafe {
(*shost).__slate_bits_0.__set_host_self_blocked(0u32);
}
scsi_run_host_queues(shost);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_unblock_requests_692: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut Scsi_Host)>, *mut ()>(Some(scsi_unblock_requests)) };

#[unsafe(no_mangle)]
extern "C" fn scsi_exit_queue() {
unsafe { kmem_cache_destroy(scsi_sense_cache) };
}

/// scsi_mode_select - issue a mode select
/// @sdev:	SCSI device to be queried
/// @pf:	Page format bit (1 == standard, 0 == vendor specific)
/// @sp:	Save page bit (0 == don't save, 1 == save)
/// @buffer: request buffer (may not be smaller than eight bytes)
/// @len:	length of request buffer.
/// @timeout: command timeout
/// @retries: number of retries before failing
/// @data: returns a structure abstracting the mode header data
/// @sshdr: place to put sense data (or NULL if no sense to be collected).
/// 	must be SCSI_SENSE_BUFFERSIZE big.
///
/// Returns zero if successful; negative error number or scsi
/// status on error
#[unsafe(no_mangle)]
extern "C" fn scsi_mode_select(sdev: *mut scsi_device, pf: i32, sp: i32, buffer: *mut u8, mut len: i32, timeout: i32, retries: i32, data: *mut scsi_mode_data, sshdr: *mut scsi_sense_hdr) -> i32 {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_9: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_11: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut cmd: [u8; 10] = [0u8; 10];
let mut real_buffer: *mut u8 = unsafe { core::mem::zeroed() };
let exec_args: scsi_exec_args = {
let mut __t0: scsi_exec_args = unsafe { core::mem::zeroed() };
__t0.sshdr = sshdr;
__t0
};
let mut ret: i32 = 0;
unsafe { memset(cmd.as_mut_ptr() as *mut (), 0i32, core::mem::size_of::<[u8; 10]>() as u64) };
cmd[1] = ((if pf != 0 { 16 } else { 0 }) | if sp != 0 { 1 } else { 0 }) as u8;
// Use MODE SELECT(10) if the device asked for it or if the mode page
// and the mode select header cannot fit within the maximumm 255 bytes
// of the MODE SELECT(6) command.
if ((unsafe { (*sdev).__slate_bits_0.__get_use_10_for_ms() }) as i32) != 0 || len.wrapping_add(4) > 255 || ((unsafe { (*data).block_descriptor_length }) as i32) > 255 {
if len > 65527i32 {
return 22i32.wrapping_neg();
}
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(8i32.wrapping_add(len) as u64, 3264u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(8i32.wrapping_add(len) as u64, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
real_buffer = __v0 as *mut u8;
if real_buffer.is_null() {
return 12i32.wrapping_neg();
}
unsafe { memcpy(real_buffer.wrapping_offset(8 as isize) as *mut (), buffer as *const (), len as u64) };
len = len.wrapping_add(8);
unsafe {
*real_buffer.wrapping_offset(0 as isize) = 0;
}
unsafe {
*real_buffer.wrapping_offset(1 as isize) = 0;
}
unsafe {
*real_buffer.wrapping_offset(2 as isize) = (*data).medium_type;
}
unsafe {
*real_buffer.wrapping_offset(3 as isize) = (*data).device_specific;
}
unsafe {
*real_buffer.wrapping_offset(4 as isize) = (if ((*data).__slate_bits_0.__get_longlba() as i32) != 0 { 1i32 } else { 0i32 }) as u8;
}
unsafe {
*real_buffer.wrapping_offset(5 as isize) = 0;
}
put_unaligned_be16(unsafe { (*data).block_descriptor_length }, real_buffer.wrapping_offset(6 as isize) as *mut ());
cmd[0] = 85;
put_unaligned_be16(len as u16, (cmd.as_mut_ptr() as *mut u8).wrapping_offset(7 as isize) as *mut ());
} else {
if ((unsafe { (*data).__slate_bits_0.__get_longlba() }) as i32) != 0 {
return 22i32.wrapping_neg();
}
let mut __v7: *mut ();
let mut __v8: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_noprof(4i32.wrapping_add(len) as u64, 3264u32, unsafe { *{
__slate_compound_9 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_noprof(4i32.wrapping_add(len) as u64, 3264u32, unsafe { *{
__slate_compound_11 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_11)
} });
}
__v8 = _res;
__v7 = __v8;
real_buffer = __v7 as *mut u8;
if real_buffer.is_null() {
return 12i32.wrapping_neg();
}
unsafe { memcpy(real_buffer.wrapping_offset(4 as isize) as *mut (), buffer as *const (), len as u64) };
len = len.wrapping_add(4);
unsafe {
*real_buffer.wrapping_offset(0 as isize) = 0;
}
unsafe {
*real_buffer.wrapping_offset(1 as isize) = (*data).medium_type;
}
unsafe {
*real_buffer.wrapping_offset(2 as isize) = (*data).device_specific;
}
unsafe {
*real_buffer.wrapping_offset(3 as isize) = (*data).block_descriptor_length as u8;
}
cmd[0] = 21;
cmd[4] = len as u8;
}
ret = scsi_execute_cmd(sdev, cmd.as_mut_ptr() as *const u8, 35u32, real_buffer as *mut (), len as u32, timeout, retries, core::ptr::addr_of!(exec_args));
unsafe { kfree(real_buffer as *const ()) };
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_mode_select_693: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, i32, i32, *mut u8, i32, i32, i32, *mut scsi_mode_data, *mut scsi_sense_hdr) -> i32>, *mut ()>(Some(scsi_mode_select)) };

/// scsi_mode_sense - issue a mode sense, falling back from 10 to six bytes if necessary.
/// @sdev:	SCSI device to be queried
/// @dbd:	set to prevent mode sense from returning block descriptors
/// @modepage: mode page being requested
/// @subpage: sub-page of the mode page being requested
/// @buffer: request buffer (may not be smaller than eight bytes)
/// @len:	length of request buffer.
/// @timeout: command timeout
/// @retries: number of retries before failing
/// @data: returns a structure abstracting the mode header data
/// @sshdr: place to put sense data (or NULL if no sense to be collected).
/// 	must be SCSI_SENSE_BUFFERSIZE big.
///
/// Returns zero if successful, or a negative error number on failure
#[unsafe(no_mangle)]
extern "C" fn scsi_mode_sense(sdev: *mut scsi_device, mut dbd: i32, modepage: i32, subpage: i32, buffer: *mut u8, len: i32, timeout: i32, retries: i32, data: *mut scsi_mode_data, mut sshdr: *mut scsi_sense_hdr) -> i32 {
let mut exec_args: core::mem::MaybeUninit<scsi_exec_args> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut scsi_exec_args = core::ptr::addr_of_mut!(exec_args) as *mut scsi_exec_args;
let mut failures: core::mem::MaybeUninit<scsi_failures> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut scsi_failures = core::ptr::addr_of_mut!(failures) as *mut scsi_failures;
let mut failure_defs: core::mem::MaybeUninit<__SlateAlign16<[scsi_failure; 2]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut [scsi_failure; 2] = core::ptr::addr_of_mut!(failure_defs) as *mut [scsi_failure; 2];
let mut my_sshdr: core::mem::MaybeUninit<scsi_sense_hdr> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut scsi_sense_hdr = core::ptr::addr_of_mut!(my_sshdr) as *mut scsi_sense_hdr;
let mut result: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(result) as *mut i32;
let mut header_length: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_6: *mut i32 = core::ptr::addr_of_mut!(header_length) as *mut i32;
let mut use_10_for_ms: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(use_10_for_ms) as *mut i32;
let mut cmd: core::mem::MaybeUninit<[u8; 12]> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut [u8; 12] = core::ptr::addr_of_mut!(cmd) as *mut [u8; 12];
unsafe {
let __slate_auto_bytes = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
let __slate_auto_bytes_1 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_6 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_3 = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_4 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
core::ptr::write(__slate_slot_3, [{
let mut __t1: scsi_failure = core::mem::zeroed();
__t1.result = 2i32;
__t1.sense = 6u8;
__t1.asc = 255u8;
__t1.ascq = 255u8;
__t1.allowed = retries as i8;
__t1
}, {
let mut __t2: scsi_failure = core::mem::zeroed();
__t2
}]);
core::ptr::write(__slate_slot_2, {
let mut __t3: scsi_failures = core::mem::zeroed();
__t3.failure_definitions = (*__slate_slot_3).as_mut_ptr() as *mut scsi_failure;
__t3
});
let __v0: *mut scsi_sense_hdr = sshdr;
*__slate_slot_1 = {
let mut __t4: scsi_exec_args = core::mem::zeroed();
__t4.sshdr = if !__v0.is_null() { __v0 } else { core::ptr::addr_of_mut!(*__slate_slot_4) };
__t4.failures = core::ptr::addr_of_mut!(*__slate_slot_2);
__t4
};
// caller might not be interested in sense, but we need it
memset(data as *mut (), 0i32, core::mem::size_of::<scsi_mode_data>() as u64);
memset(((*__slate_slot_8).as_mut_ptr() as *mut u8).wrapping_offset(0 as isize) as *mut (), 0i32, 12u64);
dbd = if ((*sdev).__slate_bits_0.__get_set_dbd_for_ms() as i32) != 0 { 8 } else { dbd };
(*__slate_slot_8)[1] = (dbd & 24) as u8; // allows DBD and LLBA bits
(*__slate_slot_8)[2] = modepage as u8;
(*__slate_slot_8)[3] = subpage as u8;
sshdr = (*__slate_slot_1).sshdr;
'__join_6: {
'__join_5: {
loop {
*__slate_slot_7 = (((*sdev).__slate_bits_0.__get_use_10_for_ms() as i32) != 0 || len > 255) as i32;
if *__slate_slot_7 != 0 {
if len < 8 || len > 65535 {
return 22i32.wrapping_neg();
} else {
(*__slate_slot_8)[0] = 90;
put_unaligned_be16(len as u16, ((*__slate_slot_8).as_mut_ptr() as *mut u8).wrapping_offset(7 as isize) as *mut ());
*__slate_slot_6 = 8;
}
} else {
if len < 4 {
break;
} else {
(*__slate_slot_8)[0] = 26;
(*__slate_slot_8)[4] = len as u8;
*__slate_slot_6 = 4;
}
}
memset(buffer as *mut (), 0i32, len as u64);
*__slate_slot_5 = scsi_execute_cmd(sdev, (*__slate_slot_8).as_mut_ptr() as *const u8, 34u32, buffer as *mut (), len as u32, timeout, retries, core::ptr::addr_of!(*__slate_slot_1));
if *__slate_slot_5 < 0 {
return *__slate_slot_5;
} else {
// This code looks awful: what it's doing is making sure an
// ILLEGAL REQUEST sense return identifies the actual command
// byte as the problem.  MODE_SENSE commands can return
// ILLEGAL REQUEST if the code page isn't supported
if !scsi_status_is_good(*__slate_slot_5) {
if scsi_sense_valid(sshdr as *const scsi_sense_hdr) {
if ((*sshdr).sense_key as i32) == 5 && ((*sshdr).asc as i32) == 32 && ((*sshdr).ascq as i32) == 0 {
// Invalid command operation code: retry using
// MODE SENSE(6) if this was a MODE SENSE(10)
// request, except if the request mode page is
// too large for MODE SENSE single byte
// allocation length field.
if *__slate_slot_7 != 0 {
if len > 255 {
return 5i32.wrapping_neg();
} else {
(*sdev).__slate_bits_0.__set_use_10_for_ms(0u32);
}
} else {
break '__join_6;
}
} else {
break '__join_6;
}
} else {
break '__join_6;
}
} else {
break '__join_5;
}
}
}
return 22i32.wrapping_neg();
}
if unlikely!((*buffer.wrapping_offset(0 as isize) as i32) == 134 && (*buffer.wrapping_offset(1 as isize) as i32) == 11 && (modepage == 6 || modepage == 8)) != 0 {
// Initio breakage?
*__slate_slot_6 = 0;
(*data).length = 13;
(*data).medium_type = 0;
(*data).device_specific = 0;
(*data).__slate_bits_0.__set_longlba(0u8);
(*data).block_descriptor_length = 0;
} else {
if *__slate_slot_7 != 0 {
(*data).length = (get_unaligned_be16(buffer.wrapping_offset(0 as isize) as *const ()) as i32).wrapping_add(2) as u32;
(*data).medium_type = *buffer.wrapping_offset(2 as isize);
(*data).device_specific = *buffer.wrapping_offset(3 as isize);
let __t0 = ((*buffer.wrapping_offset(4 as isize) as i32) & 1) as u8;
(*data).__slate_bits_0.__set_longlba(__t0);
(*data).block_descriptor_length = get_unaligned_be16(buffer.wrapping_offset(6 as isize) as *const ());
} else {
(*data).length = (*buffer.wrapping_offset(0 as isize) as i32).wrapping_add(1) as u32;
(*data).medium_type = *buffer.wrapping_offset(1 as isize);
(*data).device_specific = *buffer.wrapping_offset(2 as isize);
(*data).block_descriptor_length = *buffer.wrapping_offset(3 as isize) as u16;
}
}
(*data).header_length = *__slate_slot_6 as u8;
return 0;
}
return 5i32.wrapping_neg();
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_mode_sense_694: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, i32, i32, i32, *mut u8, i32, i32, i32, *mut scsi_mode_data, *mut scsi_sense_hdr) -> i32>, *mut ()>(Some(scsi_mode_sense)) };

/// scsi_test_unit_ready - test if unit is ready
/// @sdev:	scsi device to change the state of.
/// @timeout: command timeout
/// @retries: number of retries before failing
/// @sshdr: outpout pointer for decoded sense information.
///
/// Returns zero if successful or an error if TUR failed.  For
/// removable media, UNIT_ATTENTION sets ->changed flag.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_test_unit_ready")]
extern "C" fn scsi_test_unit_ready(sdev: *mut scsi_device, timeout: i32, mut retries: i32, sshdr: *mut scsi_sense_hdr) -> i32 {
let mut cmd: [u8; 6] = [0, 0, 0, 0, 0, 0];
let exec_args: scsi_exec_args = {
let mut __t0: scsi_exec_args = unsafe { core::mem::zeroed() };
__t0.sshdr = sshdr;
__t0
};
let mut result: i32 = 0;
// try to eat the UNIT_ATTENTION if there are enough retries
loop {
result = scsi_execute_cmd(sdev, cmd.as_mut_ptr() as *const u8, 34u32, core::ptr::null_mut::<()>(), 0u32, timeout, 1i32, core::ptr::addr_of!(exec_args));
let __v0: bool = ((unsafe { (*sdev).__slate_bits_0.__get_removable() }) as i32) != 0 && result > 0 && scsi_sense_valid(sshdr as *const scsi_sense_hdr);
if __v0 && ((unsafe { (*sshdr).sense_key }) as i32) == 6 {
unsafe {
(*sdev).__slate_bits_0.__set_changed(1u32);
}
}
let __v2: bool = result > 0 && scsi_sense_valid(sshdr as *const scsi_sense_hdr);
let __v3: bool = if __v2 && ((unsafe { (*sshdr).sense_key }) as i32) == 6 {
let __v5: i32 = retries.wrapping_sub(1);
retries = __v5;
__v5 != 0
} else {
false
};
if !__v3 {
break;
}
}
result
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_test_unit_ready_695: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, i32, i32, *mut scsi_sense_hdr) -> i32>, *mut ()>(Some(scsi_test_unit_ready)) };

/// scsi_device_set_state - Take the given device through the device state model.
/// @sdev:	scsi device to change the state of.
/// @state:	state to change to.
///
/// Returns zero if successful or an error if the requested
/// transition is illegal.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_device_set_state")]
extern "C" fn scsi_device_set_state(sdev: *mut scsi_device, state: u32) -> i32 {
let mut oldstate: u32 = unsafe { core::mem::zeroed() };
unsafe {
oldstate = (*sdev).sdev_state;
if (state as u32) == (oldstate as u32) {
return 0;
} else {
'__join_1: {
'__join_0: {
let __t8: u32 = state as u32;
if __t8 == 1 {
let __t7: u32 = oldstate as u32;
if __t7 == 9 {
break '__join_1;
} else {
break '__join_0;
}
} else {
if __t8 == 2 {
let __t6: u32 = oldstate as u32;
if __t6 == 1 {
break '__join_1;
} else {
if __t6 == 6 {
break '__join_1;
} else {
if __t6 == 7 {
break '__join_1;
} else {
if __t6 == 5 {
break '__join_1;
} else {
if __t6 == 8 {
break '__join_1;
} else {
break '__join_0;
}
}
}
}
}
} else {
if __t8 == 5 {
let __t5: u32 = oldstate as u32;
if __t5 == 2 {
break '__join_1;
} else {
if __t5 == 6 {
break '__join_1;
} else {
if __t5 == 7 {
break '__join_1;
} else {
break '__join_0;
}
}
}
} else {
if __t8 != 6 {
if __t8 != 7 {
if __t8 == 8 {
let __t3: u32 = oldstate as u32;
if __t3 == 2 {
break '__join_1;
} else {
if __t3 == 9 {
break '__join_1;
} else {
if __t3 == 5 {
break '__join_1;
} else {
if __t3 == 6 {
break '__join_1;
} else {
break '__join_0;
}
}
}
}
} else {
if __t8 == 9 {
let __t2: u32 = oldstate as u32;
if __t2 == 1 {
break '__join_1;
} else {
break '__join_0;
}
} else {
if __t8 == 3 {
let __t1: u32 = oldstate as u32;
if __t1 == 1 {
break '__join_1;
} else {
if __t1 == 2 {
break '__join_1;
} else {
if __t1 == 5 {
break '__join_1;
} else {
if __t1 == 6 {
break '__join_1;
} else {
if __t1 == 7 {
break '__join_1;
} else {
break '__join_0;
}
}
}
}
}
} else {
if __t8 == 4 {
let __t0: u32 = oldstate as u32;
if __t0 == 1 {
break '__join_1;
} else {
if __t0 == 2 {
break '__join_1;
} else {
if __t0 == 6 {
break '__join_1;
} else {
if __t0 == 7 {
break '__join_1;
} else {
if __t0 == 3 {
break '__join_1;
} else {
if __t0 == 8 {
break '__join_1;
} else {
if __t0 == 9 {
break '__join_1;
} else {
break '__join_0;
}
}
}
}
}
}
}
} else {
break '__join_1;
}
}
}
}
}
}
}
}
}
let __t4: u32 = oldstate as u32;
if __t4 == 1 {
break '__join_1;
} else {
if __t4 == 2 {
break '__join_1;
} else {
if __t4 == 5 {
break '__join_1;
} else {
if __t4 == 8 {
break '__join_1;
}
}
}
}
}
SCSI_LOG_ERROR_RECOVERY!();
return 22i32.wrapping_neg();
}
(*sdev).offline_already = false;
(*sdev).sdev_state = state;
return 0;
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_device_set_state_696: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, u32) -> i32>, *mut ()>(Some(scsi_device_set_state)) };

/// scsi_evt_emit - emit a single SCSI device uevent
/// @sdev: associated SCSI device
/// @evt: event to emit
///
/// Send a single uevent (scsi_event) to the associated scsi_device.
fn scsi_evt_emit(sdev: *mut scsi_device, evt: *mut scsi_event) {
let mut idx: i32 = 0;
let mut envp: __SlateAlign16<[*mut u8; 3]> = __SlateAlign16([0 as *mut u8; 3]);
'__slate_break_2: {
match (unsafe { (*evt).evt_type }) as u32 {
1 => {
let __v0: i32 = idx;
idx = __v0.wrapping_add(1);
envp.0[__v0 as usize] = c"SDEV_MEDIA_CHANGE=1".as_ptr() as *mut u8;
}
2 => {
unsafe { scsi_rescan_device(sdev) };
let __v3: i32 = idx;
idx = __v3.wrapping_add(1);
envp.0[__v3 as usize] = c"SDEV_UA=INQUIRY_DATA_HAS_CHANGED".as_ptr() as *mut u8;
}
3 => {
let __v5: i32 = idx;
idx = __v5.wrapping_add(1);
envp.0[__v5 as usize] = c"SDEV_UA=CAPACITY_DATA_HAS_CHANGED".as_ptr() as *mut u8;
}
4 => {
let __v7: i32 = idx;
idx = __v7.wrapping_add(1);
envp.0[__v7 as usize] = c"SDEV_UA=THIN_PROVISIONING_SOFT_THRESHOLD_REACHED".as_ptr() as *mut u8;
}
5 => {
let __v9: i32 = idx;
idx = __v9.wrapping_add(1);
envp.0[__v9 as usize] = c"SDEV_UA=MODE_PARAMETERS_CHANGED".as_ptr() as *mut u8;
}
6 => {
let __v11: i32 = idx;
idx = __v11.wrapping_add(1);
envp.0[__v11 as usize] = c"SDEV_UA=REPORTED_LUNS_DATA_HAS_CHANGED".as_ptr() as *mut u8;
}
7 => {
let __v13: i32 = idx;
idx = __v13.wrapping_add(1);
envp.0[__v13 as usize] = c"SDEV_UA=ASYMMETRIC_ACCESS_STATE_CHANGED".as_ptr() as *mut u8;
}
8 => {
let __v15: i32 = idx;
idx = __v15.wrapping_add(1);
envp.0[__v15 as usize] = c"SDEV_UA=POWER_ON_RESET_OCCURRED".as_ptr() as *mut u8;
}
_ => {
break '__slate_break_2; // do nothing
}
}
}
let __v17: i32 = idx;
idx = __v17.wrapping_add(1);
envp.0[__v17 as usize] = core::ptr::null_mut::<u8>();
unsafe { kobject_uevent_env(core::ptr::addr_of_mut!((*sdev).sdev_gendev.kobj), 2u32, envp.0.as_mut_ptr() as *mut *mut u8) };
}

/// scsi_evt_thread - send a uevent for each scsi event
/// @work: work struct for scsi_device
///
/// Dispatch queued events to their associated scsi_device kobjects
/// as uevents.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_evt_thread")]
extern "C" fn scsi_evt_thread(work: *mut work_struct) {
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut evt_type: u32 = 1;
let mut event_list: list_head = unsafe { core::mem::zeroed() };
event_list = list_head { next: core::ptr::addr_of_mut!(event_list), prev: core::ptr::addr_of_mut!(event_list) };
sdev = container_of__24!(work, scsi_device);
while (evt_type as u32) <= 8 {
if test_and_clear_bit((evt_type as u32) as i64, (unsafe { (*sdev).pending_events.as_mut_ptr() as *mut u64 }) as *mut u64) {
sdev_evt_send_simple(sdev, evt_type, ((BIT!(10i32) | BIT!(11i32)) as u32) | (BIT!(6i32) as u32) | (BIT!(7i32) as u32));
}
evt_type = (evt_type as u32).wrapping_add(1) as u32;
}
while true {
let mut evt: *mut scsi_event = unsafe { core::mem::zeroed() };
let mut this: *mut list_head = unsafe { core::mem::zeroed() };
let mut tmp: *mut list_head = unsafe { core::mem::zeroed() };
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sdev).list_lock) }, flags);
list_splice_init(unsafe { core::ptr::addr_of_mut!((*sdev).event_list) }, core::ptr::addr_of_mut!(event_list));
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*sdev).list_lock) }, flags);
if list_empty(core::ptr::addr_of_mut!(event_list) as *const list_head) != 0 {
break;
}
this = unsafe { (*core::ptr::addr_of_mut!(event_list)).next };
tmp = unsafe { (*this).next };
while list_is_head(this as *const list_head, core::ptr::addr_of_mut!(event_list) as *const list_head) == 0 {
evt = list_entry__2!(this, scsi_event);
list_del(unsafe { core::ptr::addr_of_mut!((*evt).node) });
scsi_evt_emit(sdev, evt);
unsafe { kfree(evt as *const ()) };
this = tmp;
tmp = unsafe { (*this).next };
}
}
}

/// sdev_evt_send - send asserted event to uevent thread
/// @sdev: scsi_device event occurred on
/// @evt: event to send
///
/// Assert scsi device event asynchronously.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.sdev_evt_send")]
extern "C" fn sdev_evt_send(sdev: *mut scsi_device, evt: *mut scsi_event) {
let mut flags: u64 = 0;
spin_lock_irqsave!(unsafe { core::ptr::addr_of_mut!((*sdev).list_lock) }, flags);
list_add_tail(unsafe { core::ptr::addr_of_mut!((*evt).node) }, unsafe { core::ptr::addr_of_mut!((*sdev).event_list) });
schedule_work(unsafe { core::ptr::addr_of_mut!((*sdev).event_work) });
spin_unlock_irqrestore(unsafe { core::ptr::addr_of_mut!((*sdev).list_lock) }, flags);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_sdev_evt_send_697: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, *mut scsi_event)>, *mut ()>(Some(sdev_evt_send)) };

/// sdev_evt_alloc - allocate a new scsi event
/// @evt_type: type of event to allocate
/// @gfpflags: GFP flags for allocation
///
/// Allocates and returns a new scsi_event.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.sdev_evt_alloc")]
extern "C" fn sdev_evt_alloc(evt_type: u32, gfpflags: u32) -> *mut scsi_event {
let mut __slate_compound_3: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut evt: *mut scsi_event = unsafe { core::mem::zeroed() };
let mut __v0: *mut scsi_event;
let __obj_size: u64 = size_mul(core::mem::size_of::<scsi_event>() as u64, 1u64);
let mut __v1: *mut ();
let mut __v2: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(__obj_size, gfpflags, unsafe { *{
__slate_compound_3 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_3)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(__obj_size, gfpflags, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
}
__v2 = _res;
__v1 = __v2;
__v0 = __v1 as *mut scsi_event;
evt = __v0;
if evt.is_null() {
return core::ptr::null_mut::<scsi_event>();
}
unsafe {
(*evt).evt_type = evt_type;
}
INIT_LIST_HEAD(unsafe { core::ptr::addr_of_mut!((*evt).node) });
// evt_type-specific initialization, if any
'__slate_break_6: {
match evt_type as u32 {
_ => {
break '__slate_break_6; // do nothing
}
}
}
evt
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_sdev_evt_alloc_698: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(u32, u32) -> *mut scsi_event>, *mut ()>(Some(sdev_evt_alloc)) };

/// sdev_evt_send_simple - send asserted event to uevent thread
/// @sdev: scsi_device event occurred on
/// @evt_type: type of event to send
/// @gfpflags: GFP flags for allocation
///
/// Assert scsi device event asynchronously, given an event type.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.sdev_evt_send_simple")]
extern "C" fn sdev_evt_send_simple(sdev: *mut scsi_device, evt_type: u32, gfpflags: u32) {
let evt: *mut scsi_event = sdev_evt_alloc(evt_type, gfpflags);
if evt.is_null() {
unsafe { sdev_prefix_printk(c"\x013".as_ptr() as *const u8, sdev as *const scsi_device, core::ptr::null::<u8>(), c"event %d eaten due to OOM\n".as_ptr() as *const u8, evt_type as u32) };
return;
}
sdev_evt_send(sdev, evt);
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_sdev_evt_send_simple_699: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, u32, u32)>, *mut ()>(Some(sdev_evt_send_simple)) };

/// scsi_device_quiesce - Block all commands except power management.
/// @sdev:	scsi device to quiesce.
///
/// This works by trying to transition to the SDEV_QUIESCE state
/// (which must be a legal transition).  When the device is in this
/// state, only power management requests will be accepted, all others will
/// be deferred.
///
/// Must be called with user context, may sleep.
///
/// Returns zero if successful or an error if not.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_device_quiesce")]
extern "C" fn scsi_device_quiesce(sdev: *mut scsi_device) -> i32 {
let q: *mut request_queue = unsafe { (*sdev).request_queue };
let mut memflags: u32 = 0;
let mut err: i32 = 0;
// It is allowed to call scsi_device_quiesce() multiple times from
// the same context but concurrent scsi_device_quiesce() calls are
// not allowed.
let mut __v0: i64;
let mut __ret_warn_on: i32 = 0;
let __v1: bool = !unsafe { (*sdev).quiesced_by }.is_null() && (unsafe { (*sdev).quiesced_by }) != get_current();
__ret_warn_on = __v1 as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("700: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 700b - ., 3; .popsection\n# {0}", const 700 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_device_quiesce_2, sym __slate_asm_string_scsi_device_quiesce_3, const 2844 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("701: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 701b - ., 4; .popsection\n# {0}", const 701 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
if (unsafe { (*sdev).quiesced_by }) == get_current() {
return 0;
}
unsafe { blk_set_pm_only(q) };
memflags = blk_mq_freeze_queue(q);
// Ensure that the effect of blk_set_pm_only() will be visible
// for percpu_ref_tryget() callers that occur after the queue
// unfreeze even if the queue was already frozen before this function
// was called. See also https://lwn.net/Articles/573497/.
unsafe { synchronize_rcu() };
blk_mq_unfreeze_queue(q, memflags);
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
err = scsi_device_set_state(sdev, 5u32);
if err == 0 {
unsafe {
(*sdev).quiesced_by = get_current();
}
} else {
unsafe { blk_clear_pm_only(q) };
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
err
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_device_quiesce_702: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device) -> i32>, *mut ()>(Some(scsi_device_quiesce)) };

/// scsi_device_resume - Restart user issued commands to a quiesced device.
/// @sdev:	scsi device to resume.
///
/// Moves the device from quiesced back to running and restarts the
/// queues.
///
/// Must be called with user context, may sleep.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_device_resume")]
extern "C" fn scsi_device_resume(sdev: *mut scsi_device) {
// check if the device state was mutated prior to resume, and if
// so assume the state is being managed elsewhere (for example
// device deleted during suspend)
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
if ((unsafe { (*sdev).sdev_state }) as u32) == 5 {
scsi_device_set_state(sdev, 2u32);
}
if !unsafe { (*sdev).quiesced_by }.is_null() {
unsafe {
(*sdev).quiesced_by = core::ptr::null_mut::<task_struct>();
}
unsafe { blk_clear_pm_only((*sdev).request_queue) };
}
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_device_resume_703: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device)>, *mut ()>(Some(scsi_device_resume)) };

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.device_quiesce_fn")]
extern "C" fn device_quiesce_fn(sdev: *mut scsi_device, data: *mut ()) {
scsi_device_quiesce(sdev);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_target_quiesce")]
extern "C" fn scsi_target_quiesce(starget: *mut scsi_target) {
unsafe { starget_for_each_device(starget, core::ptr::null_mut::<()>(), Some(device_quiesce_fn)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_target_quiesce_704: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_target)>, *mut ()>(Some(scsi_target_quiesce)) };

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.device_resume_fn")]
extern "C" fn device_resume_fn(sdev: *mut scsi_device, data: *mut ()) {
scsi_device_resume(sdev);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_target_resume")]
extern "C" fn scsi_target_resume(starget: *mut scsi_target) {
unsafe { starget_for_each_device(starget, core::ptr::null_mut::<()>(), Some(device_resume_fn)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_target_resume_705: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_target)>, *mut ()>(Some(scsi_target_resume)) };

fn __scsi_internal_device_block_nowait(sdev: *mut scsi_device) -> i32 {
if scsi_device_set_state(sdev, 8u32) != 0 {
return scsi_device_set_state(sdev, 9u32);
}
0
}

#[unsafe(no_mangle)]
extern "C" fn scsi_start_queue(sdev: *mut scsi_device) {
let mut __v0: u32;
let __ai_ptr: *mut u32 = unsafe { core::ptr::addr_of_mut!((*sdev).queue_stopped) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u32>() as u64);
let mut __v2: u32;
let mut __ret: u32 = 0;
let mut __old: u32 = 1;
let mut __new: u32 = 0;
match (const { core::mem::size_of::<u32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v2 = __ret;
__v0 = __v2;
if __v0 != 0 {
unsafe { blk_mq_unquiesce_queue((*sdev).request_queue) };
}
}

fn scsi_stop_queue(sdev: *mut scsi_device) {
// The atomic variable of ->queue_stopped covers that
// blk_mq_quiesce_queue* is balanced with blk_mq_unquiesce_queue.
//
// The caller needs to wait until quiesce is done.
let mut __v0: u32;
let __ai_ptr: *mut u32 = unsafe { core::ptr::addr_of_mut!((*sdev).queue_stopped) };
kcsan_mb!();
instrument_atomic_read_write(__ai_ptr as *const (), core::mem::size_of::<u32>() as u64);
let mut __v2: u32;
let mut __ret: u32 = 0;
let mut __old: u32 = 0;
let mut __new: u32 = 1;
match (const { core::mem::size_of::<u32>() }) as u64 {
1 => {
let mut __ptr: *mut u8 = __ai_ptr as *mut u8;
unsafe {
let __t0: u32;
core::arch::asm!("lock cmpxchgb {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t0, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
}
2 => {
let mut __ptr: *mut u16 = __ai_ptr as *mut u16;
unsafe {
let __t1: u32;
core::arch::asm!("lock cmpxchgw {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t1, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t1 as u32) };
}
}
4 => {
let mut __ptr: *mut u32 = __ai_ptr as *mut u32;
unsafe {
let __t2: u32;
core::arch::asm!("lock cmpxchgl {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t2, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t2 as u32) };
}
}
8 => {
let mut __ptr: *mut u64 = __ai_ptr as *mut u64;
unsafe {
let __t3: u32;
core::arch::asm!("lock cmpxchgq {1:e}, ({0:r})", in(reg) __ptr, in(reg) __new as u32, inlateout("ax") __old as u32 => __t3, options(nostack, att_syntax));
__ret = unsafe { core::mem::transmute::<u32, u32>(__t3 as u32) };
}
}
_ => {
unsafe { __cmpxchg_wrong_size() };
}
}
__v2 = __ret;
__v0 = __v2;
if __v0 == 0 {
unsafe { blk_mq_quiesce_queue_nowait((*sdev).request_queue) };
}
}

/// scsi_internal_device_block_nowait - try to transition to the SDEV_BLOCK state
/// @sdev: device to block
///
/// Pause SCSI command processing on the specified device. Does not sleep.
///
/// Returns zero if successful or a negative error code upon failure.
///
/// Notes:
/// This routine transitions the device to the SDEV_BLOCK state (which must be
/// a legal transition). When the device is in this state, command processing
/// is paused until the device leaves the SDEV_BLOCK state. See also
/// scsi_internal_device_unblock_nowait().
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_internal_device_block_nowait")]
extern "C" fn scsi_internal_device_block_nowait(sdev: *mut scsi_device) -> i32 {
let ret: i32 = __scsi_internal_device_block_nowait(sdev);
// The device has transitioned to SDEV_BLOCK.  Stop the
// block layer from calling the midlayer with this device's
// request queue.
if ret == 0 {
scsi_stop_queue(sdev);
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_internal_device_block_nowait_706: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device) -> i32>, *mut ()>(Some(scsi_internal_device_block_nowait)) };

/// scsi_device_block - try to transition to the SDEV_BLOCK state
/// @sdev: device to block
/// @data: dummy argument, ignored
///
/// Pause SCSI command processing on the specified device. Callers must wait
/// until all ongoing scsi_queue_rq() calls have finished after this function
/// returns.
///
/// Note:
/// This routine transitions the device to the SDEV_BLOCK state (which must be
/// a legal transition). When the device is in this state, command processing
/// is paused until the device leaves the SDEV_BLOCK state. See also
/// scsi_internal_device_unblock().
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_device_block")]
extern "C" fn scsi_device_block(sdev: *mut scsi_device, data: *mut ()) {
let mut err: i32 = 0;
let mut state: u32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
err = __scsi_internal_device_block_nowait(sdev);
state = unsafe { (*sdev).sdev_state };
if err == 0 {
scsi_stop_queue(sdev);
}
// scsi_stop_queue() must be called with the state_mutex
// held. Otherwise a simultaneous scsi_start_queue() call
// might unquiesce the queue before we quiesce it.
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
let mut __v0: i32;
let __ret_warn_on: i32 = (err != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
2i32 | BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"%s: failed to block %s in state %d\n".as_ptr() as *const u8, c"scsi_device_block".as_ptr() as *mut u8, dev_name(core::ptr::addr_of_mut!((*sdev).sdev_gendev) as *const device), state as u32) };
let mut __v1: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v1 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v2: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_scsi_device_block_3, sym __slate_asm_string_scsi_device_block_4, const 3014 as i32, const 2323 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v2 = bug;
unsafe { __v1.unwrap()(__v2, c"scsi_device_block".as_ptr() as *mut u8, dev_name(core::ptr::addr_of_mut!((*sdev).sdev_gendev) as *const device), state as u32) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v0 = __ret_warn_on;
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_707: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// scsi_internal_device_unblock_nowait - resume a device after a block request
/// @sdev:	device to resume
/// @new_state:	state to set the device to after unblocking
///
/// Restart the device queue for a previously suspended SCSI device. Does not
/// sleep.
///
/// Returns zero if successful or a negative error code upon failure.
///
/// Notes:
/// This routine transitions the device to the SDEV_RUNNING state or to one of
/// the offline states (which must be a legal transition) allowing the midlayer
/// to goose the queue for this device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_internal_device_unblock_nowait")]
extern "C" fn scsi_internal_device_unblock_nowait(sdev: *mut scsi_device, new_state: u32) -> i32 {
match new_state as u32 {
2 | 7 => {
}
_ => {
return 22i32.wrapping_neg();
}
}
// Try to transition the scsi device to SDEV_RUNNING or one of the
// offlined states and goose the device queue if successful.
match (unsafe { (*sdev).sdev_state }) as u32 {
7 | 8 => {
unsafe {
(*sdev).sdev_state = new_state;
}
}
9 => {
if (new_state as u32) == 7 || (new_state as u32) == 6 {
unsafe {
(*sdev).sdev_state = new_state;
}
} else {
unsafe {
(*sdev).sdev_state = 1;
}
}
}
3 | 6 => {
}
_ => {
return 22i32.wrapping_neg();
}
}
scsi_start_queue(sdev);
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_internal_device_unblock_nowait_708: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, u32) -> i32>, *mut ()>(Some(scsi_internal_device_unblock_nowait)) };

/// scsi_internal_device_unblock - resume a device after a block request
/// @sdev:	device to resume
/// @new_state:	state to set the device to after unblocking
///
/// Restart the device queue for a previously suspended SCSI device. May sleep.
///
/// Returns zero if successful or a negative error code upon failure.
///
/// Notes:
/// This routine transitions the device to the SDEV_RUNNING state or to one of
/// the offline states (which must be a legal transition) allowing the midlayer
/// to goose the queue for this device.
fn scsi_internal_device_unblock(sdev: *mut scsi_device, new_state: u32) -> i32 {
let mut ret: i32 = 0;
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
ret = scsi_internal_device_unblock_nowait(sdev, new_state);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
ret
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.target_block")]
extern "C" fn target_block(dev: *mut device, data: *mut ()) -> i32 {
if (unsafe { scsi_is_target_device(dev as *const device) }) != 0 {
unsafe { starget_for_each_device(to_scsi_target!(dev), core::ptr::null_mut::<()>(), Some(scsi_device_block)) };
}
0
}

/// scsi_block_targets - transition all SCSI child devices to SDEV_BLOCK state
/// @dev: a parent device of one or more scsi_target devices
/// @shost: the Scsi_Host to which this device belongs
///
/// Iterate over all children of @dev, which should be scsi_target devices,
/// and switch all subordinate scsi devices to SDEV_BLOCK state. Wait for
/// ongoing scsi_queue_rq() calls to finish. May sleep.
///
/// Note:
/// @dev must not itself be a scsi_target device.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_block_targets")]
extern "C" fn scsi_block_targets(shost: *mut Scsi_Host, dev: *mut device) {
let mut __v0: i64;
let __ret_warn_on: i32 = ((unsafe { scsi_is_target_device(dev as *const device) }) != 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("709: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 709b - ., 3; .popsection\n# {0}", const 709 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_block_targets_1, sym __slate_asm_string_scsi_block_targets_2, const 3121 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("710: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 710b - ., 4; .popsection\n# {0}", const 710 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
unsafe { device_for_each_child(dev, core::ptr::null_mut::<()>(), Some(target_block)) };
unsafe { blk_mq_wait_quiesce_done(core::ptr::addr_of_mut!((*shost).tag_set)) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_block_targets_711: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut Scsi_Host, *mut device)>, *mut ()>(Some(scsi_block_targets)) };

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.device_unblock")]
extern "C" fn device_unblock(sdev: *mut scsi_device, data: *mut ()) {
scsi_internal_device_unblock(sdev, unsafe { *(data as *mut u32) });
}

#[unsafe(link_section = ".text.slate_distinct.scsi_lib.target_unblock")]
extern "C" fn target_unblock(dev: *mut device, data: *mut ()) -> i32 {
if (unsafe { scsi_is_target_device(dev as *const device) }) != 0 {
unsafe { starget_for_each_device(to_scsi_target!(dev), data, Some(device_unblock)) };
}
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_target_unblock")]
extern "C" fn scsi_target_unblock(dev: *mut device, mut new_state: u32) {
if (unsafe { scsi_is_target_device(dev as *const device) }) != 0 {
unsafe { starget_for_each_device(to_scsi_target!(dev), core::ptr::addr_of_mut!(new_state) as *mut (), Some(device_unblock)) };
} else {
unsafe { device_for_each_child(dev, core::ptr::addr_of_mut!(new_state) as *mut (), Some(target_unblock)) };
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_target_unblock_712: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut device, u32)>, *mut ()>(Some(scsi_target_unblock)) };

/// scsi_host_block - Try to transition all logical units to the SDEV_BLOCK state
/// @shost: device to block
///
/// Pause SCSI command processing for all logical units associated with the SCSI
/// host and wait until pending scsi_queue_rq() calls have finished.
///
/// Returns zero if successful or a negative error code upon failure.
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_host_block")]
extern "C" fn scsi_host_block(shost: *mut Scsi_Host) -> i32 {
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
// Call scsi_internal_device_block_nowait so we can avoid
// calling synchronize_rcu() for each LUN.
sdev = unsafe { __scsi_iterate_devices(shost, core::ptr::null_mut::<scsi_device>()) };
while !sdev.is_null() {
unsafe { mutex_lock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
ret = scsi_internal_device_block_nowait(sdev);
unsafe { mutex_unlock(core::ptr::addr_of_mut!((*sdev).state_mutex)) };
if ret != 0 {
unsafe { scsi_device_put(sdev) };
return ret;
}
sdev = unsafe { __scsi_iterate_devices(shost, sdev) };
}
// Wait for ongoing scsi_queue_rq() calls to finish.
unsafe { blk_mq_wait_quiesce_done(core::ptr::addr_of_mut!((*shost).tag_set)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_host_block_713: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut Scsi_Host) -> i32>, *mut ()>(Some(scsi_host_block)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_host_unblock")]
extern "C" fn scsi_host_unblock(shost: *mut Scsi_Host, new_state: i32) -> i32 {
let mut sdev: *mut scsi_device = unsafe { core::mem::zeroed() };
let mut ret: i32 = 0;
sdev = unsafe { __scsi_iterate_devices(shost, core::ptr::null_mut::<scsi_device>()) };
while !sdev.is_null() {
ret = scsi_internal_device_unblock(sdev, new_state as u32);
if ret != 0 {
unsafe { scsi_device_put(sdev) };
break;
}
sdev = unsafe { __scsi_iterate_devices(shost, sdev) };
}
ret
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_host_unblock_714: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut Scsi_Host, i32) -> i32>, *mut ()>(Some(scsi_host_unblock)) };

/// scsi_kmap_atomic_sg - find and atomically map an sg-elemnt
/// @sgl:	scatter-gather list
/// @sg_count:	number of segments in sg
/// @offset:	offset in bytes into sg, on return offset into the mapped area
/// @len:	bytes to map, on return number of bytes mapped
///
/// Returns virtual address of the start of the mapped page
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_kmap_atomic_sg")]
extern "C" fn scsi_kmap_atomic_sg(__v15: *mut scatterlist, sg_count: i32, offset: *mut u64, len: *mut u64) -> *mut () {
let mut i: i32 = 0;
let mut sg_len: u64 = 0;
let mut len_complete: u64 = 0;
let mut sg: *mut scatterlist = unsafe { core::mem::zeroed() };
let mut __v0: i64;
{
let __ret_warn_on: i32 = (irqs_disabled!() == 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("715: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 715b - ., 3; .popsection\n# {0}", const 715 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_kmap_atomic_sg_11, sym __slate_asm_string_scsi_kmap_atomic_sg_12, const 3223 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
__v0 = (__ret_warn_on != 0) as i64;
}
sg = __v15;
while i < sg_count {
len_complete = sg_len; // Complete sg-entries
sg_len = sg_len.wrapping_add((unsafe { (*sg).length }) as u64);
if sg_len > unsafe { *offset } {
break;
}
i = i.wrapping_add(1);
sg = sg_next(sg);
}
if unlikely!(i == sg_count) != 0 {
let mut __v22: i32;
while false {
}
__v22 = unsafe { _printk(c"\x013%s: Bytes in sg: %zu, requested offset %zu, elements %d\n".as_ptr() as *const u8, c"scsi_kmap_atomic_sg".as_ptr() as *mut u8, sg_len, *offset, sg_count) };
let mut __v24: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("717: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 717b - ., 3; .popsection\n# {0}", const 717 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_scsi_kmap_atomic_sg_25, sym __slate_asm_string_scsi_kmap_atomic_sg_26, const 3236 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
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
return core::ptr::null_mut::<()>();
}
// Offset starting from the beginning of first page in this sg-entry
unsafe {
*offset = unsafe { *offset }.wrapping_sub(len_complete).wrapping_add((*sg).offset as u64);
}
let page: *mut page = sg_page(sg).wrapping_offset(((unsafe { *offset }) >> 12i32) as isize);
let __v30: u64 = unsafe { *offset };
let __v31: u64 = __v30 & !(!(_AC!() << 12i32).wrapping_sub(1));
unsafe {
*offset = __v31;
}
// Bytes in this sg-entry from *offset to the end of the page
sg_len = (_AC!() << 12i32).wrapping_sub(unsafe { *offset });
if (unsafe { *len }) > sg_len {
unsafe {
*len = sg_len;
}
}
kmap_atomic(page as *const page)
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_kmap_atomic_sg_719: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scatterlist, i32, *mut u64, *mut u64) -> *mut ()>, *mut ()>(Some(scsi_kmap_atomic_sg)) };

// scsi_kunmap_atomic_sg - atomically unmap a virtual address, previously mapped with scsi_kmap_atomic_sg
// @virt:	virtual address to be unmapped
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_kunmap_atomic_sg_721: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut ())>, *mut ()>(Some(scsi_kunmap_atomic_sg)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.sdev_disable_disk_events")]
extern "C" fn sdev_disable_disk_events(sdev: *mut scsi_device) {
atomic_inc(unsafe { core::ptr::addr_of_mut!((*sdev).disk_events_disable_depth) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_sdev_disable_disk_events_722: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device)>, *mut ()>(Some(sdev_disable_disk_events)) };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.sdev_enable_disk_events")]
extern "C" fn sdev_enable_disk_events(sdev: *mut scsi_device) {
let mut __v0: i64;
let __ret_warn_on: i32 = (atomic_read((unsafe { core::ptr::addr_of_mut!((*sdev).disk_events_disable_depth) }) as *const __SlateRecord_types_h_190) <= 0) as i32;
if unlikely__2!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("723: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 723b - ., 3; .popsection\n# {0}", const 723 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_sdev_enable_disk_events_1, sym __slate_asm_string_sdev_enable_disk_events_2, const 3273 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("724: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 724b - ., 4; .popsection\n# {0}", const 724 as i32, options(nostack, att_syntax));
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
atomic_dec(unsafe { core::ptr::addr_of_mut!((*sdev).disk_events_disable_depth) });
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_sdev_enable_disk_events_725: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device)>, *mut ()>(Some(sdev_enable_disk_events)) };

fn designator_prio(d: *const u8) -> u8 {
if ((unsafe { *d.wrapping_offset(1 as isize) }) as i32) & 48 != 0 {
return 0;
}
// not associated with LUN
if ((unsafe { *d.wrapping_offset(3 as isize) }) as i32) == 0 {
return 0;
}
// invalid length
// Order of preference for lun descriptor:
// - SCSI name string
// - NAA IEEE Registered Extended
// - EUI-64 based 16-byte
// - EUI-64 based 12-byte
// - NAA IEEE Registered
// - NAA IEEE Extended
// - EUI-64 based 8-byte
// - SCSI name string (truncated)
// - T10 Vendor ID
// as longer descriptors reduce the likelyhood
// of identification clashes.
match ((unsafe { *d.wrapping_offset(1 as isize) }) as i32) & 15 {
8 => {
return 9; // SCSI name string, variable-length UTF-8
}
3 => {
match ((unsafe { *d.wrapping_offset(4 as isize) }) as i32) >> 4i32 {
6 => {
return 8; // NAA registered extended
}
5 => {
return 5; // NAA registered
}
4 => {
return 4; // NAA extended
}
3 => {
return 1; // NAA locally assigned
}
_ => {
}
}
}
2 => {
match (unsafe { *d.wrapping_offset(3 as isize) }) as i32 {
16 => {
return 7; // EUI64-based, 16 byte
}
12 => {
return 6; // EUI64-based, 12 byte
}
8 => {
return 3; // EUI64-based, 8 byte
}
_ => {
}
}
}
1 => {
return 1; // T10 vendor ID
}
_ => {
}
}
0
}

// scsi_vpd_lun_id - return a unique device identification
// @sdev: SCSI device
// @id:   buffer for the identification
// @id_len:  length of the buffer
//
// Copies a unique device identification into @id based
// on the information in the VPD page 0x83 of the device.
// The string will be formatted as a SCSI name string.
//
// Returns the length of the identification or error on failure.
// If the identifier is longer than the supplied buffer the actual
// identifier length is returned and the buffer is not zero-padded.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_vpd_lun_id_728: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, *mut u8, u64) -> i32>, *mut ()>(Some(scsi_vpd_lun_id)) };

// scsi_vpd_lun_serial - return a unique device serial number
// @sdev: SCSI device
// @sn:   buffer for the serial number
// @sn_size: size of the buffer
//
// Copies the device serial number into @sn based on the information in
// the VPD page 0x80 of the device. The string will be null terminated
// and have leading and trailing whitespace stripped.
//
// Returns the length of the serial number or error on failure.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_vpd_lun_serial_732: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, *mut u8, u64) -> i32>, *mut ()>(Some(scsi_vpd_lun_serial)) };

// scsi_vpd_tpg_id - return a target port group identifier
// @sdev: SCSI device
// @rel_id: pointer to return relative target port in if not %NULL
//
// Returns the Target Port Group identifier from the information
// from VPD page 0x83 of the device.
// Optionally sets @rel_id to the relative target port on success.
//
// Return: the identifier or error on failure.
#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_vpd_tpg_id_735: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_device, *mut i32) -> i32>, *mut ()>(Some(scsi_vpd_tpg_id)) };

/// scsi_build_sense - build sense data for a command
/// @scmd:	scsi command for which the sense should be formatted
/// @desc:	Sense format (non-zero == descriptor format,
///              0 == fixed format)
/// @key:	Sense key
/// @asc:	Additional sense code
/// @ascq:	Additional sense code qualifier
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.scsi_lib.scsi_build_sense")]
extern "C" fn scsi_build_sense(scmd: *mut scsi_cmnd, desc: i32, key: u8, asc: u8, ascq: u8) {
unsafe { scsi_build_sense_buffer(desc, (*scmd).sense_buffer, key, asc, ascq) };
unsafe {
(*scmd).result = 2;
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_scsi_build_sense_736: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(*mut scsi_cmnd, i32, u8, u8, u8)>, *mut ()>(Some(scsi_build_sense)) };

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_dma_set_max_seg_size_1: [u8; 1] = *b"\0";

static __slate_asm_string_dma_set_max_seg_size_2: [u8; 57] = *b"/home/takashi/c-corpus/linux/include/linux/dma-mapping.h\0";

static __slate_asm_string_dma_set_seg_boundary_1: [u8; 1] = *b"\0";

static __slate_asm_string_dma_set_seg_boundary_2: [u8; 57] = *b"/home/takashi/c-corpus/linux/include/linux/dma-mapping.h\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_scsi_alloc_sgtables_29: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_alloc_sgtables_30: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_alloc_sgtables_31: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_alloc_sgtables_32: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_alloc_sgtables_33: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_alloc_sgtables_34: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_block_targets_1: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_block_targets_2: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_complete_1: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_complete_2: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_device_block_3: [u8; 36] = *b"%s: failed to block %s in state %d\n\0";

static __slate_asm_string_scsi_device_block_4: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_device_quiesce_2: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_device_quiesce_3: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_device_state_check_2: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_device_state_check_3: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_end_request_1: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_end_request_2: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_execute_cmd_12: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_execute_cmd_13: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_init_limits_8: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_init_limits_9: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_io_completion_5: [u8; 47] = *b"Bytes remaining after failed, no-retry command\0";

static __slate_asm_string_scsi_io_completion_6: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_kmap_atomic_sg_11: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_kmap_atomic_sg_12: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_kmap_atomic_sg_25: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_kmap_atomic_sg_26: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_mq_requeue_cmd_4: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_mq_requeue_cmd_5: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_queue_rq_22: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_queue_rq_23: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_rq_err_bytes_3: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_rq_err_bytes_4: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_scsi_setup_scsi_cmnd_0: [u8; 1] = *b"\0";

static __slate_asm_string_scsi_setup_scsi_cmnd_1: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_sdev_enable_disk_events_1: [u8; 1] = *b"\0";

static __slate_asm_string_sdev_enable_disk_events_2: [u8; 53] = *b"/home/takashi/c-corpus/linux/drivers/scsi/scsi_lib.c\0";

static __slate_asm_string_sg_assign_page_0: [u8; 1] = *b"\0";

static __slate_asm_string_sg_assign_page_1: [u8; 57] = *b"/home/takashi/c-corpus/linux/include/linux/scatterlist.h\0";

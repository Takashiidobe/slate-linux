
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
//! 	Intel IO-APIC support for multi-Pentium hosts.
//!
//! 	Copyright (C) 1997, 1998, 1999, 2000, 2009 Ingo Molnar, Hajnalka Szabo
//!
//! 	Many thanks to Stig Venaas for trying out countless experimental
//! 	patches and reporting/debugging problems patiently!
//!
//! 	(c) 1999, Multiple IO-APIC support, developed by
//! 	Ken-ichi Yaku <yaku@css1.kbnes.nec.co.jp> and
//!      Hidemi Kishimoto <kisimoto@css1.kbnes.nec.co.jp>,
//! 	further tested and cleaned up by Zach Brown <zab@redhat.com>
//! 	and Ingo Molnar <mingo@redhat.com>
//!
//! 	Fixes
//! 	Maciej W. Rozycki	:	Bits for genuine 82489DX APICs;
//! 					thanks to Eric Gilmore
//! 					and Rolf G. Tews
//! 					for testing these extensively
//! 	Paul Diefenbaugh	:	Added full ACPI support
//!
//! Historical information which is worth to be preserved:
//!
//! - SiS APIC rmw bug:
//!
//! 	We used to have a workaround for a bug in SiS chips which
//! 	required to rewrite the index register for a read-modify-write
//! 	operation as the chip lost the index information which was
//! 	setup for the read already. We cache the data now, so that
//! 	workaround has been removed.
//! This counts to 15. Any more, it will return 16th argument.
//! Operation not permitted
//! This error code is special: arch syscall entry code will return
//! -ENOSYS if users try to call a syscall that doesn't exist.  To keep
//! failures of syscalls that really do exist distinguishable from
//! failures due to attempts to use a nonexistent syscall, syscall
//! implementations should refrain from returning -ENOSYS.
//! restart by calling sys_restart_syscall
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
//! Internal kernel timer frequency
//! casts are necessary for constants, because we never know how for sure
//! how U/UL/ULL map to __u16, __u32, __u64. At least not in a portable way.
//! issue num suppressed message on exit
//! strlen("ratelimit") + 1
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
//! label
//! Defines x86 CPU feature bits
//! 64bit FS base
//! Needs two entries
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
//! upper_32_bits - return bits 32-63 of a number
//! @n: the number we're accessing
//!
//! A basic shift-right of a 64- or 32-bit quantity.  Use this to suppress
//! the "right shift count >= width of type" warning when that quantity is
//! 32-bits.
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
//! The size of each LDT entry.
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
//! This function is needed by all SMP systems. It must _always_ be valid
//! from the initial startup.
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
//! This is the IO-APIC register space as specified
//! by Intel docs:
//! Each PCI slot may be a combo card with its own bus.  4 IRQ pins per slot.
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
//! Macros to add or remove encryption attribute
//! Must be a compile-time constant, so implement it as a macro
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
//! Bits set in the VMA until the stack is in its final location
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
//! Number of standard BARs
//! Number of reset methods used in pci_reset_fn_methods array in pci.c
//! These macros should be used after a dma_map_sg call has been done
//! to get bus addresses of each of the SG entries and their lengths.
//! You should only work with the number of sg entries dma_map_sg
//! returns, or alternatively stop on the first sg_dma_len(sg) which
//! is 0.
//! DMA_ATTR_WRITE_COMBINE: Specifies that writes to the mapping may be
//! buffered to improve performance.
//! These are specified by iBCS2
//! Epoll event masks
//! register details
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
//! 8237 DMA controllers
//! We are using top down, so it is safe to use 0 here
//! Macro to enforce the same ordering and stack sizes
//! i8259A PIC related value
//! Monochrome Text Display
//! number of edd_info structs starting at EDDBUF
//! This is the maximum number of entries in struct boot_params::e820_table
//! (the zeropage), which is part of the x86 boot protocol ABI:
//! Currently, the max observed size in the kernel code is
//! JUMP_LABEL_NOP_SIZE/RELATIVEJUMP_SIZE, which are 5.
//! Raise it if needed.
//! Define the default level of output to be very little This can be turned
//! up by using apic=verbose for more information and apic=debug for _lots_
//! of information.  apic_verbosity is defined in apic.c
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
// Concatenate two parameters, but allow them to be expanded beforehand.
// No such file or directory
// I/O error
// No such device or address
// Argument list too long
// Try again
// Out of memory
// Bad address
// Device or resource busy
// File exists
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
// Operation is not supported
// DECLARE_FLEX_ARRAY() - Declare a flexible array usable in a union
//
// @TYPE: The type of each flexible array element
// @NAME: The name of the flexible array member
//
// In order to have a flexible array member in a union or alone in a
// struct, it needs to be wrapped in an anonymous struct with at least 1
// named member, but that member can be empty.
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
// Combine multiple asm inline constraint args into a single arg for passing to
// another macro.
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
// informational
// __swahw32 - return a word-swapped 32-bit value
// @x: value to wordswap
//
// __swahw32(0x12340000) is 0x00001234
// __swahb32 - return a high and low byte-swapped 32-bit value
// @x: value to byteswap
//
// __swahb32(0x12345678) is 0x34127856
// FW_BUG
// Add this to a message where you are sure the firmware is buggy or behaves
// really stupid or out of spec. Be aware that the responsible BIOS developer
// should be able to fix this issue or at least get a concrete idea of the
// problem by reading your message without the need of looking at the kernel
// code.
//
// Use it for definite and high priority BIOS bugs.
//
// FW_WARN
// Use it for not that clear (e.g. could the kernel messed up things already?)
// and medium priority BIOS bugs.
//
// FW_INFO
// Use this one if you want to tell the user or vendor about something
// suspicious, but generally harmless related to the firmware.
//
// Use it for information or very low priority BIOS bugs.
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
// pr_cont - Continues a previous log message in the same line.
// @fmt: format string
// @...: arguments for the format string
//
// This macro expands to a printk with KERN_CONT loglevel. It should only be
// used when continuing a log message with no newline ('\n') enclosed. Otherwise
// it defaults back to KERN_DEFAULT loglevel.
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
// Overflow Flag
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
// "apic" Onboard APIC
// "acpi" ACPI via MSR
// "sse"
// "rdtscp" RDTSCP
// "rep_good" REP microcode works well
// "monitor" MONITOR/MWAIT support
// "est" Enhanced SpeedStep
// "x2apic" X2APIC
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
// "11ap" Bad local APIC aka 11AP
// "l1tf" CPU is affected by L1 Terminal Fault
// "div0" AMD DIV0 speculation bug
// Needs two entries
// Number of entries in the GDT table:
// Bit size and mask of CPU number stored in the per CPU data (and TSC_AUX)
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
// Atomic operations are already serializing on x86
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
// lower_32_bits - return bits 0-31 of a number
// @n: the number we're accessing
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
// Docs say 0 for future compat.
// GFP_ZONE_BAD is a bitmap for all combinations of __GFP_DMA, __GFP_DMA32
// __GFP_HIGHMEM and __GFP_MOVABLE that are not permitted. One flag per
// entry starting with bit 0. Bit is set if the combination is not
// allowed.
// mprotect needs to preserve PAT and encryption bits when updating
// vm_page_prot
// Currently stuck as a macro due to indirect forward reference to
// linux/mmzone.h's __section_mem_map_addr() definition:
// All top-level MITIGATION_PAGE_TABLE_ISOLATION page tables are order-1 pages
// (8k-aligned and 8k in size).  The kernel one is at the beginning 4k and
// the user one is in the last 4k.  To switch between them, you
// just need to flip the 12th bit in their addresses.
// a shortcut which implies the use of the kernel's pgd, instead
// of a process's
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
// We overload the LSB of the page pointer to indicate whether it's
// a valid sg entry, or whether it points to the start of a new scatterlist.
// Those low bits are there for everyone! (thanks mason :-)
// These helpers provide future and backwards compatibility
// for accessing popular PCI BAR info
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
// Periodic intr. / Square wave rate select. 0=none, 1=32.8kHz,... 15=2Hz
// periodic interrupt enable
// Flexible array members are not allowed to be part of a union under
// C99, but this is not for any technical reason. Work around the
// limitation.
// Fixed by ACPI spec
// DWORD 5
// 8 bit slave DMA, channels 0..3
// 16 bit master DMA, ch 4(=slave input)..7
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// single-channel mask (w)
// mode register (w)
// clear pointer flip-flop (w)
// DMA page registers
// CGA Display
// EGA/VGA in Monochrome Mode
// EGA in Color Mode
// VGA+ in Color Mode
// VESA VGA in graphic mode
// ACER PICA-61 local S3 video
// MIPS Magnum 4000 G364 video
// Various SGI graphics hardware
// DEC TGA
// Sun frame buffer.
// Sun PCI based frame buffer.
// PowerMacintosh frame buffer.
// EFI graphic mode
// Frame buffer base is 64-bit
// max number of signatures to store
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
macro_rules! ALIGN__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL__2!($x, $a)
};
}

macro_rules! BITS_PER_TYPE {
($type:ty) => {
(core::mem::size_of::<$type>() as u64).wrapping_mul(8i32 as $type)
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

/// The yet supported machines all access the RTC index register via
/// an ISA port access but the way to access the date register differs ...
macro_rules! CMOS_READ {
($addr:expr) => {
unsafe { rtc_cmos_read(($addr) as u8) }
};
}

macro_rules! CMOS_WRITE {
($val:expr, $addr:expr) => {
unsafe { rtc_cmos_write(($val) as u8, ($addr) as u8) };
};
}

macro_rules! CMOS_WRITE__2 {
($val:expr, $addr:expr) => {
unsafe { rtc_cmos_write($val, ($addr) as u8) };
};
}

macro_rules! EAX_EDX_VAL {
($low:expr, $high:expr) => {
($low) | ($high) << 32i32
};
}

macro_rules! GENMASK_INPUT_CHECK__20 {
() => {
core::mem::size_of::<__SlateRecord_find_h_386>() as i32
};
}

/// Generate a mask for the specified type @t. Additional checks are made to
/// guarantee the value returned fits in that type, relying on
/// -Wshift-count-overflow compiler check to detect incompatible arguments.
/// For example, all these create build errors or warnings:
///
/// - GENMASK(15, 20): wrong argument order
/// - GENMASK(72, 15): doesn't fit unsigned long
/// - GENMASK_U32(33, 15): doesn't fit in a u32
macro_rules! GENMASK_TYPE__4 {
($t:ty, $h:expr, $l:expr) => {
(GENMASK_INPUT_CHECK__20!() as $t).wrapping_add(type_max!() << $l & type_max!() >> BITS_PER_TYPE!($t).wrapping_sub(1i32 as $t).wrapping_sub($h))
};
}

macro_rules! GENMASK__11 {
($h:expr, $l:expr) => {
GENMASK_TYPE__4!(u64, $h, $l)
};
}

macro_rules! IO_APIC_IRQ {
($x:expr) => {
($x) >= 16 || ((1i32 << $x) as u64) & unsafe { io_apic_irqs } != 0
};
}

macro_rules! IO_APIC_IRQ__2 {
($x:expr) => {
($x) >= 16 || ((1i32 << $x) as u64) & unsafe { io_apic_irqs } != 0
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

macro_rules! PASS_BUCKET_PARAMS {
($_size:expr) => {
$_size
};
}

macro_rules! PASS_KMALLOC_PARAMS {
($_size:expr) => {
PASS_BUCKET_PARAMS!($_size)
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

macro_rules! REQUIRED_MASK_BIT_SET {
($x:expr) => {
(if ($x) >> 5i32 == 0 { 117481833 } else { if ($x) >> 5i32 == 1 { 536870912 } else { if ($x) >> 5i32 == 3 { 36732928 } else { 0 } } }) & 1u32 << (($x) & 31)
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
18446744071562067968u64
};
}

macro_rules! _AC__3 {
() => {
16045481047390945280u64
};
}

macro_rules! _AT {
($T:ty, $X:expr) => {
($X) as $T
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

macro_rules! __ALIGN_KERNEL_MASK {
($x:expr, $mask:expr) => {
$x.wrapping_add($mask) & !$mask
};
}

macro_rules! __ALIGN_KERNEL__2 {
($x:expr, $a:expr) => {
__ALIGN_KERNEL_MASK!($x, (($a) as u32).wrapping_sub(1))
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

macro_rules! __fix_to_virt__3 {
($x:expr) => {
round_up__2!((10u64.wrapping_neg() << 20i32).wrapping_add(4096u64), 2097152i32).wrapping_sub(_AC!() << 12i32).wrapping_sub(($x) << 12i32)
};
}

macro_rules! __fix_to_virt {
($x:expr) => {
round_up__2!((10u64.wrapping_neg() << 20i32).wrapping_add(4096u64), 2097152i32).wrapping_sub(_AC!() << 12i32).wrapping_sub((($x) << 12i32) as u64)
};
}

macro_rules! __irqd_to_state {
($d:expr) => {
unsafe { (*(*$d).common).state_use_accessors }
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
macro_rules! __native_word__4 {
() => {
((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<u8>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i16>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i32>() }) as u64) || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)
};
}

/// Is this type a native word size -- useful for atomic operations
macro_rules! __native_word__2 {
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

/// This looks more complex than it should be. But we need to
/// get the type for the ~ right in round_down (it needs to be
/// as wide as the result!), and we want to evaluate the macro
/// arguments just once each.
macro_rules! __round_mask__2 {
($y:expr) => {
$y.wrapping_sub(1) as u64
};
}

/// The following macros are non-atomic versions of their non-underscored
/// counterparts.
macro_rules! __set_bit__4 {
($nr:expr, $addr:expr) => {
if false && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
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
macro_rules! __sign_use__4 {
($ux:expr) => {
if is_signed_type__3!() { 2i32.wrapping_add(__is_nonneg!($ux) as i32) } else { 1i32.wrapping_add(2i32.wrapping_mul((((const { core::mem::size_of::<u32>() }) as u64) < 4) as i32)) }
};
}

/// We need to compute the minimum and maximum values representable in a given
/// type. These macros may also be useful elsewhere. It would seem more obvious
/// to do something like:
///
/// #define type_min(T) (T)(is_signed_type(T) ? (T)1 << (8*sizeof(T)-1) : 0)
/// #define type_max(T) (T)(is_signed_type(T) ? ((T)1 << (8*sizeof(T)-1)) - 1 : ~(T)0)
///
/// Unfortunately, the middle expressions, strictly speaking, have
/// undefined behaviour, and at least some versions of gcc warn about
/// the type_max expression (but not if -fsanitize=undefined is in
/// effect; in that case, the warning is deferred to runtime...).
///
/// The slightly excessive casting in type_min is to make sure the
/// macros also produce sensible values for the exotic type _Bool. [The
/// overflow checkers only almost work for _Bool, but that's
/// a-feature-not-a-bug, since people shouldn't be doing arithmetic on
/// _Bools. Besides, the gcc builtins don't allow _Bool* as third
/// argument.]
///
/// Idea stolen from
/// https://mail-index.netbsd.org/tech-misc/2007/02/05/0000.html -
/// credit to Christian Biere.
macro_rules! __type_half_max {
() => {
1u64 << 8u64.wrapping_mul(core::mem::size_of::<u64>() as u64).wrapping_sub(1).wrapping_sub((is_signed_type!() as i32) as u64)
};
}

macro_rules! __type_max {
() => {
__type_half_max!().wrapping_sub(1).wrapping_add(__type_half_max!())
};
}

macro_rules! __types_ok__4 {
($ux:expr, $uy:expr) => {
__sign_use__4!($ux) & __sign_use__4!($uy)
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
macro_rules! __verify_pcpu_ptr__9 {
() => {
loop {
let __v: *const () = core::ptr::null_mut::<i32>() as *const ();
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

macro_rules! bitmap_size {
($nbits:expr) => {
ALIGN__2!($nbits, 64i32) / 8
};
}

macro_rules! boot_cpu_has {
($bit:expr) => {
cpu_has!(unsafe { core::ptr::addr_of_mut!(boot_cpu_data) }, $bit)
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
macro_rules! container_of__32 {
($ptr:expr) => {
{
let mut __v: *mut irq_pin_list;
__v = ((unsafe { ((($ptr) as *mut ()) as *const u8).wrapping_offset(-(offsetof__37!() as isize)) }) as *mut ()) as *mut irq_pin_list;
__v
}
};
}

macro_rules! cpu_has {
($c:expr, $bit:expr) => {
{
let __v: i32 = if true && REQUIRED_MASK_BIT_SET!($bit) != 0 {
1
} else {
test_cpu_cap!($c, $bit) as i32
};
__v
}
};
}

/// ffz - find first zero bit in word
/// @word: The word to search
///
/// Undefined if no zero exists, so code should check against ~0UL first.
macro_rules! ffz {
($word:expr) => {
if unsafe { __slate_intrinsic_f217c4720f04f6db($word) } { ((!$word).trailing_zeros() as i32) as u64 } else { variable_ffz($word) }
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
macro_rules! is_signed_type {
() => {
(1i32.wrapping_neg() as u64) < 1
};
}

macro_rules! kcsan_check_atomic_read {
($ptr:expr, $size:expr) => {
kcsan_check_access($ptr, $size, 4i32);
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

/// list_entry - get the struct for this entry
/// @ptr:	the &struct list_head pointer.
/// @type:	the type of the struct this is embedded in.
/// @member:	the name of the list_head within the struct.
macro_rules! list_entry__17 {
($ptr:expr) => {
container_of__32!($ptr)
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
list_entry__17!(unsafe { (*$ptr).next })
};
}

/// list_next_entry - get the next element in list
/// @pos:	the type * to cursor
/// @member:	the name of the list_head within the struct.
macro_rules! list_next_entry__2 {
($pos:expr, $member:ident) => {
list_entry__17!(unsafe { (*$pos).$member.next })
};
}

macro_rules! local_irq_disable {
() => {
loop {
raw_local_irq_disable!();
if true {
break;
}
}
};
}

macro_rules! local_irq_enable {
() => {
loop {
raw_local_irq_enable!();
if true {
break;
}
}
};
}

macro_rules! local_irq_restore {
($flags:expr) => {
loop {
raw_local_irq_restore!($flags);
if true {
break;
}
}
};
}

macro_rules! local_irq_save {
($flags:expr) => {
loop {
raw_local_irq_save!($flags);
if true {
break;
}
}
};
}

/// mdelay - Inserting a delay based on milliseconds with busy waiting
/// @n:	requested delay in milliseconds
///
/// See udelay() for basic information about mdelay() and it's variants.
///
/// Please double check, whether mdelay() is the right way to go or whether a
/// refactoring of the code is the better variant to be able to use msleep()
/// instead.
macro_rules! mdelay {
($n:expr) => {
if true && ($n) <= 5 {
udelay($n.wrapping_mul(1000) as u64);
} else {
let mut __ms: u64 = ($n) as u64;
loop {
let __v: u64 = __ms;
__ms = __v.wrapping_sub(1);
if __v == 0 {
break;
}
udelay(1000u64);
}
}
};
}

macro_rules! memblock_alloc_or_panic {
($size:expr, $align:expr) => {
unsafe { __memblock_alloc_or_panic($size, ($align) as u64, c"ioapic_setup_resources".as_ptr() as *const u8) }
};
}

macro_rules! memblock_alloc_or_panic__3 {
($size:expr, $align:expr) => {
unsafe { __memblock_alloc_or_panic($size, $align, c"io_apic_init_mappings".as_ptr() as *const u8) }
};
}

macro_rules! mpc_ioapic_ver {
($ioapic_idx:expr) => {
unsafe { ioapics.0[($ioapic_idx) as usize].mp_config.apicver }
};
}

macro_rules! offsetof__37 {
() => {
core::mem::offset_of!(irq_pin_list, list) as u64
};
}

macro_rules! per_cpu__2 {
($var:expr, $cpu:expr) => {
unsafe { *per_cpu_ptr__3!(core::ptr::addr_of_mut!($var), $cpu) }
};
}

macro_rules! per_cpu_offset {
($x:expr) => {
unsafe { __per_cpu_offset[($x) as usize] }
};
}

macro_rules! per_cpu_ptr__3 {
($ptr:expr, $cpu:expr) => {
{
let mut __v: *mut i32;
__verify_pcpu_ptr__9!();
__v = (((($ptr) as u64) as *mut i32) as u64).wrapping_add(per_cpu_offset!($cpu)) as *mut i32;
__v
}
};
}

macro_rules! pgprot_val {
($x:expr) => {
$x.pgprot
};
}

macro_rules! raw_check_bogus_irq_restore {
() => {
while false {
}
};
}

/// Wrap the arch provided IRQ routines to provide appropriate checks.
macro_rules! raw_local_irq_disable {
() => {
arch_local_irq_disable();
};
}

macro_rules! raw_local_irq_enable {
() => {
arch_local_irq_enable();
};
}

macro_rules! raw_local_irq_restore {
($flags:expr) => {
loop {
typecheck!(u64);
raw_check_bogus_irq_restore!();
arch_local_irq_restore($flags);
if true {
break;
}
}
};
}

macro_rules! raw_local_irq_save {
($flags:expr) => {
loop {
typecheck!(u64);
$flags = arch_local_irq_save();
if true {
break;
}
}
};
}

macro_rules! raw_spin_lock {
($lock:expr) => {
unsafe { _raw_spin_lock($lock) };
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

macro_rules! raw_spin_unlock {
($lock:expr) => {
unsafe { _raw_spin_unlock($lock) };
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

/// round_up - round up to next specified power of 2
/// @x: the value to round
/// @y: multiple to round up to (must be a power of 2)
///
/// Rounds @x up to next multiple of @y (which must be a power of 2).
/// To perform arbitrary rounding up, use roundup() below.
macro_rules! round_up__2 {
($x:expr, $y:expr) => {
($x.wrapping_sub(1) | __round_mask__2!($y)).wrapping_add(1)
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits__2 {
($nbits:expr) => {
(unsafe { __slate_intrinsic_1c5364fcd9da51fe($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
};
}

/// small_const_nbits(n) is true precisely when it is known at compile-time
/// that BITMAP_SIZE(n) is 1, i.e. 1 <= n <= BITS_PER_LONG. This allows
/// various bit/bitmap APIs to provide a fast inline implementation. Bitmaps
/// of size 0 are very rare, and a compile-time-known-size 0 is most likely
/// a sign of error. They will be handled correctly by the bit/bitmap APIs,
/// but using the out-of-line functions, so that the inline implementations
/// can unconditionally dereference the pointer(s).
macro_rules! small_const_nbits {
($nbits:expr) => {
(unsafe { __slate_intrinsic_f217c4720f04f6db($nbits) }) && ($nbits) <= 64 && ($nbits) > 0
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
let __v: i32 = if (unsafe { __slate_intrinsic_a39e427d3b6c2df4($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_bit__5 {
($nr:expr, $addr:expr) => {
{
let __v: i32 = if (unsafe { __slate_intrinsic_70b71f185ebb8c4a($nr) }) && unsafe { __slate_intrinsic_1cfe0293d0a2af8d((($addr) as u64) != 0) } && (($addr) as u64) != 0 && unsafe { __slate_intrinsic_f217c4720f04f6db(*(($addr) as *const u64)) } {
const_test_bit(($nr) as u64, ($addr) as *const u64) as i32
} else {
_test_bit(($nr) as u64, ($addr) as *const u64) as i32
};
__v
}
};
}

macro_rules! test_cpu_cap {
($c:expr, $bit:expr) => {
arch_test_bit(($bit) as u64, ((unsafe { (*$c).__slate_anon_9.x86_capability.as_mut_ptr() as *mut u32 }) as *mut u64) as *const u64)
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

macro_rules! type_max {
() => {
__type_max!()
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
    static mut __SCK__WARN_trap: static_call_key;
    static mut kmsg_fops: file_operations;
    static mut __per_cpu_offset: [u64; 64];
    static mut __default_kernel_pte_mask: u64;
    static mut current_task: *mut task_struct;
    static mut const_current_task: *mut task_struct;
    static mut phys_base: u64;
    static mut page_offset_base: u64;
    static mut boot_cpu_data: cpuinfo_x86;
    static mut cpuinfo_op: seq_operations;
    static mut __tracepoint_read_msr: tracepoint;
    static mut jiffies: u64;
    static mut x86_init: x86_init_ops;
    static mut x86_platform: x86_platform_ops;
    static mut x86_apic_ops: x86_apic_ops;
    static mut smp_found_config: i32;
    static mut numa_node: i32;
    static mut iomem_resource: resource;
    static mut kmalloc_caches: [[*mut kmem_cache; 14]; 3];
    static mut no_irq_chip: irq_chip;
    static mut noioapicquirk: i32;
    static mut noioapicreroute: i32;
    static mut x86_vector_domain: *mut irq_domain;
    static mut global_clock_event: *mut clock_event_device;
    static mut legacy_pic: *mut legacy_pic;
    static mut io_apic_irqs: u64;
    static mut apic_verbosity: i32;
    static mut apic_is_disabled: bool;
    static mut apic: *mut apic;
    static mut __SCK__apic_call_eoi: static_call_key;
    static mut __SCK__apic_call_read: static_call_key;
    static mut __SCK__apic_call_write: static_call_key;
    fn __WARN_trap(arg0: *mut bug_entry, ...);
    fn __SCT__WARN_trap(arg0: *mut bug_entry, ...);
    fn panic(arg0: *const u8, ...);
    fn __builtin_expect(arg0: i64, arg1: i64) -> i64;
    fn _printk(arg0: *const u8, ...) -> i32;
    fn __bad_size_call_parameter();
    fn ____wrong_branch_error() -> bool;
    fn cachemode2protval(arg0: u32) -> u64;
    fn _find_first_zero_bit(arg0: *const u64, arg1: u64) -> u64;
    fn memcpy(arg0: *mut (), arg1: *const (), arg2: u64) -> *mut ();
    fn memset(arg0: *mut (), arg1: i32, arg2: u64) -> *mut ();
    fn memcmp(arg0: *const (), arg1: *const (), arg2: u64) -> i32;
    fn snprintf(arg0: *mut u8, arg1: u64, arg2: *const u8, ...) -> i32;
    fn __compiletime_assert_16();
    fn __compiletime_assert_17();
    fn __compiletime_assert_18();
    fn __compiletime_assert_21();
    fn __compiletime_assert_22();
    fn irq_get_nr_irqs() -> u32;
    fn irq_to_desc(arg0: u32) -> *mut irq_desc;
    fn irq_get_next_irq(arg0: u32) -> u32;
    fn __compiletime_assert_57();
    fn _raw_spin_lock(arg0: *mut raw_spinlock);
    fn _raw_spin_lock_irqsave(arg0: *mut raw_spinlock) -> u64;
    fn _raw_spin_unlock(arg0: *mut raw_spinlock);
    fn _raw_spin_unlock_irqrestore(arg0: *mut raw_spinlock, arg1: u64);
    fn mutex_lock(arg0: *mut mutex);
    fn mutex_unlock(arg0: *mut mutex);
    fn do_trace_read_msr(arg0: u32, arg1: u64, arg2: i32);
    fn native_set_fixmap(arg0: u32, arg1: u64, arg2: pgprot);
    fn release_resource(arg0: *mut resource) -> i32;
    fn insert_resource(arg0: *mut resource, arg1: *mut resource) -> i32;
    fn kfree(arg0: *const ());
    fn __compiletime_assert_257();
    fn __builtin_unreachable();
    fn __kmalloc_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn __kmalloc_cache_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: u64) -> *mut ();
    fn __kmalloc_cache_node_noprof(arg0: *mut kmem_cache, arg1: u32, arg2: i32, arg3: u64) -> *mut ();
    fn __kmalloc_large_noprof(arg0: u64, arg1: u32) -> *mut ();
    fn __kmalloc_large_node_noprof(arg0: u64, arg1: u32, arg2: i32) -> *mut ();
    fn cache_get_priv_group(arg0: *mut cacheinfo) -> *const attribute_group;
    fn hrtimer_start_range_ns(arg0: *mut hrtimer, arg1: i64, arg2: u64, arg3: u32);
    fn __compiletime_assert_376();
    fn __compiletime_assert_419();
    fn irq_stat_inc_and_enable(arg0: u32);
    fn irq_cfg(arg0: u32) -> *mut irq_cfg;
    fn irqd_cfg(arg0: *mut irq_data) -> *mut irq_cfg;
    fn irq_complete_move(arg0: *mut irq_cfg);
    fn irq_move_masked_irq(arg0: *mut irq_data);
    fn handle_fasteoi_irq(arg0: *mut irq_desc);
    fn handle_edge_irq(arg0: *mut irq_desc);
    fn irq_chip_compose_msi_msg(arg0: *mut irq_data, arg1: *mut msi_msg) -> i32;
    fn irq_chip_ack_parent(arg0: *mut irq_data);
    fn irq_chip_retrigger_hierarchy(arg0: *mut irq_data) -> i32;
    fn irq_set_chip_and_handler_name(arg0: u32, arg1: *const irq_chip, arg2: Option<unsafe extern "C" fn(*mut irq_desc)>, arg3: *const u8);
    fn __irq_set_handler(arg0: u32, arg1: Option<unsafe extern "C" fn(*mut irq_desc)>, arg2: i32, arg3: *const u8);
    fn irq_modify_status(arg0: u32, arg1: u64, arg2: u64);
    fn irq_set_chip(arg0: u32, arg1: *const irq_chip) -> i32;
    fn irq_get_irq_data(arg0: u32) -> *mut irq_data;
    fn __bad_udelay();
    fn __udelay(arg0: u64);
    fn __const_udelay(arg0: u64);
    fn __delay(arg0: u64);
    fn rtc_cmos_read(arg0: u8) -> u8;
    fn rtc_cmos_write(arg0: u8, arg1: u8);
    fn register_syscore(arg0: *mut syscore);
    fn __memblock_alloc_or_panic(arg0: u64, arg1: u64, arg2: *const u8) -> *mut ();
    fn __irq_domain_alloc_fwnode(arg0: u32, arg1: i32, arg2: *const u8, arg3: *mut u64, arg4: *mut fwnode_handle) -> *mut fwnode_handle;
    fn irq_domain_free_fwnode(arg0: *mut fwnode_handle);
    fn irq_domain_instantiate(arg0: *const irq_domain_info) -> *mut irq_domain;
    fn irq_find_matching_fwspec(arg0: *mut irq_fwspec, arg1: u32) -> *mut irq_domain;
    fn irq_domain_remove(arg0: *mut irq_domain);
    fn __irq_resolve_mapping(arg0: *mut irq_domain, arg1: u64, arg2: *mut u32) -> *mut irq_desc;
    fn irq_domain_get_irq_data(arg0: *mut irq_domain, arg1: u32) -> *mut irq_data;
    fn __irq_domain_alloc_irqs(arg0: *mut irq_domain, arg1: i32, arg2: u32, arg3: i32, arg4: *mut (), arg5: bool, arg6: *const irq_affinity_desc) -> i32;
    fn irq_domain_free_irqs(arg0: u32, arg1: u32);
    fn irq_domain_activate_irq(arg0: *mut irq_data, arg1: bool) -> i32;
    fn irq_domain_deactivate_irq(arg0: *mut irq_data);
    fn irq_domain_free_irqs_top(arg0: *mut irq_domain, arg1: u32, arg2: u32);
    fn irq_domain_alloc_irqs_parent(arg0: *mut irq_domain, arg1: u32, arg2: u32, arg3: *mut ()) -> i32;
    fn irq_domain_free_irqs_parent(arg0: *mut irq_domain, arg1: u32, arg2: u32);
    fn init_irq_alloc_info(arg0: *mut irq_alloc_info, arg1: *const cpumask);
    fn copy_irq_alloc_info(arg0: *mut irq_alloc_info, arg1: *mut irq_alloc_info);
    fn panic_if_irq_remap(arg0: *const u8);
    fn disconnect_bsp_APIC(arg0: i32);
    fn sync_Arb_IDs();
    fn __SCT__apic_call_eoi();
    fn __SCT__apic_call_read(arg0: u32) -> u32;
    fn __SCT__apic_call_write(arg0: u32, arg1: u32);
    fn apic_ack_irq(arg0: *mut irq_data);
    fn __compiletime_assert_658();
}

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.is.constant.i32"]
    fn __slate_intrinsic_1c5364fcd9da51fe(arg0: u32) -> bool;
    #[link_name = "llvm.is.constant.i1"]
    fn __slate_intrinsic_1cfe0293d0a2af8d(arg0: bool) -> bool;
    #[link_name = "llvm.is.constant.i8"]
    fn __slate_intrinsic_70b71f185ebb8c4a(arg0: u8) -> bool;
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
// Format: <modname>__<counter>_<line>_<fn>
// Format: __<prefix>__<iid><id>
#[repr(C)]
#[derive(Clone, Copy)]
struct obs_kernel_param {
    str: *const u8,
    setup_func: Option<unsafe extern "C" fn(*mut u8) -> i32>,
    early: i32,
}


// Only for really core code.  See moduleparam.h for the normal way.
//
// Force the alignment so the compiler doesn't space elements of the
// obs_kernel_param "array" too far apart in .init.setup.
// NOTE: __setup functions return values:
// @fn returns 1 (or non-zero) if the option argument is "handled"
// and returns 0 if the option argument is "not handled".
// NOTE: @fn is as per module_param, not __setup!
// I.e., @fn returns 0 for no error or non-zero for error
// (possibly @fn returns a -errno value, but it does not matter).
// Emits warning if @fn returns non-zero.
#[inline(always)]
fn __must_check_overflow(overflow: bool) -> bool {
unlikely__2!(overflow) != 0
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


#[inline(always)]
fn IS_ERR(ptr: *const ()) -> bool {
IS_ERR_VALUE!(ptr as u64) != 0
}

fn cc_platform_has(attr: u32) -> bool {
false
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
core::arch::asm!("1: jmp {2} # objtool NOPs this \n\t.pushsection __jump_table,  \"aw\" \n\t .balign 8 \n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n.long 1b - . \n\t.long {2} - . \n\t .quad  {0}+8 + {1} + 2 - . \n\t.popsection \n\t", sym __tracepoint_read_msr, const 0 as u8, label {
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
union __SlateRecord_jump_label_h_95 {
    r#type: u64,
    entries: *mut jump_entry,
    next: *mut static_key_mod,
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
struct static_key_mod {
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
    __slate_bits_0: __slate_bits::__SlateBits106U0,
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
fn native_irq_disable() {
unsafe {
core::arch::asm!("cli", options(nostack, att_syntax));
}
}

#[inline(always)]
fn native_irq_enable() {
unsafe {
core::arch::asm!("sti", options(nostack, att_syntax));
}
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct desc_struct {
    limit0: u16,
    base0: u16,
    __slate_bits_0: __slate_bits::__SlateBits107U0,
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


#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct __SlateRecord_instrumented_h_66 {
    x: i64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_find_h_386 {
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
if unlikely!(__ret_warn_on) != 0 {
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

// instrument_get_user() - add instrumentation to get_user()-like macros
// @to: destination variable, may not be address-taken
//
// get_user() and friends are fragile, so it may depend on the implementation
// whether the instrumentation happens before or after the data is copied from
// the userspace.
#[inline(always)]
fn generic___set_bit(nr: u64, addr: *mut u64) {
let mask: u64 = BIT_MASK!(nr);
let p: *mut u64 = (addr as *mut u64).wrapping_offset(BIT_WORD!(nr) as isize);
unsafe {
*p = *p | mask;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_bitmap {
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
fn arch___set_bit(mut nr: u64, mut addr: *mut u64) {
unsafe {
core::arch::asm!(" btsq  {1:r},({0:r})", in(reg) addr as *mut i64, in(reg) nr as u64, options(nostack, att_syntax));
}
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

#[inline(always)]
fn variable__ffs(mut word: u64) -> u64 {
unsafe {
let __t0: u64;
core::arch::asm!("tzcnt {1:r},{0:r}", lateout(reg) __t0, in(reg) word as u64, options(pure, nomem, nostack, att_syntax));
word = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
}
word
}

// __ffs - find first set bit in word
// @word: The word to search
//
// Undefined if no bit exists, so code should check against 0 first.
#[inline(always)]
fn variable_ffz(word: u64) -> u64 {
variable__ffs(!word)
}

#[inline(always)]
fn ___set_bit(nr: u64, addr: *mut u64) {
instrument_write(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch___set_bit(nr, addr);
}

#[inline(always)]
fn _test_bit(nr: u64, addr: *const u64) -> bool {
instrument_atomic_read(addr.wrapping_offset(BIT_WORD!(nr) as isize) as *const (), core::mem::size_of::<i64>() as u64);
arch_test_bit(nr, addr)
}

#[inline(always)]
fn find_first_zero_bit(addr: *const u64, size: u64) -> u64 {
if small_const_nbits!(size) {
let val: u64 = (unsafe { *addr }) | !GENMASK__11!(size.wrapping_sub(1), 0i32);
return if val == !0 { size } else { ffz!(val) };
}
unsafe { _find_first_zero_bit(addr, size) }
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_111 {
    hw_cpu_type: u32,
    __slate_anon_1: __SlateRecord_processor_h_113,
    __slate_anon_2: __SlateRecord_processor_h_119,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits153U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_119 {
    __slate_bits_0: __slate_bits::__SlateBits154U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_136 {
    __slate_anon_0: __SlateRecord_processor_h_142,
    x86_vfm: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_processor_h_142 {
    x86_model: u8,
    x86: u8,
    x86_vendor: u8,
    x86_reserved: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_154 {
    intel_platform_id: u8,
    amd_unused: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_processor_h_177 {
    x86_capability: [u32; 24],
    x86_capability_alignment: u64,
}


#[inline(always)]
fn bitmap_zero(dst: *mut u64, nbits: u32) {
let len: u32 = bitmap_size!(nbits);
if small_const_nbits__2!(nbits) {
unsafe {
*dst = 0;
}
} else {
unsafe { memset(dst as *mut (), 0i32, len as u64) };
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct perf_event {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_smp_types_h_60 {
    u_flags: u32,
    a_flags: __SlateRecord_types_h_190,
}


fn __list_add_valid(new: *mut list_head, prev: *mut list_head, next: *mut list_head) -> bool {
true
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
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

fn __list_del(prev: *mut list_head, next: *mut list_head) {
unsafe {
(*next).prev = prev;
}
loop {
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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


fn __list_del_entry(entry: *mut list_head) {
if !__list_del_entry_valid(entry) {
return;
}
__list_del(unsafe { (*entry).prev }, unsafe { (*entry).next });
}

fn list_del(entry: *mut list_head) {
__list_del_entry(entry);
unsafe {
(*entry).next = ((256i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
unsafe {
(*entry).prev = ((290i32 as *const u8).wrapping_offset(_AC__3!() as isize) as *mut ()) as *mut list_head;
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_spinlock_types_h_18 {
    rlock: raw_spinlock,
}


fn list_is_head(list: *const list_head, head: *const list_head) -> i32 {
(list == head) as i32
}

fn list_empty(head: *const list_head) -> i32 {
let mut __v0: *mut list_head;
loop {
if !(__native_word__4!() || ((const { core::mem::size_of::<*mut list_head>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
unsafe { __compiletime_assert_22() };
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
// list_for_each_entry_safe - iterate over list of given type safe against removal of list entry
// @pos:	the type * to use as a loop cursor.
// @n:		another type * to use as temporary storage
// @head:	the head for your list.
// @member:	the name of the list_head within the struct.
#[inline(always)]
fn arch_local_save_flags() -> u64 {
native_save_fl()
}

#[inline(always)]
fn arch_local_irq_disable() {
native_irq_disable();
}

#[inline(always)]
fn arch_local_irq_enable() {
native_irq_enable();
}

#[inline(always)]
fn arch_local_irq_save() -> u64 {
let flags: u64 = arch_local_save_flags();
arch_local_irq_disable();
flags
}

#[inline(always)]
fn arch_irqs_disabled_flags(flags: u64) -> i32 {
(flags & _BITUL!(9i32) == 0) as i32
}

#[inline(always)]
fn arch_local_irq_restore(flags: u64) {
if arch_irqs_disabled_flags(flags) == 0 {
arch_local_irq_enable();
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct fred_cs {
    __slate_bits_0: __slate_bits::__SlateBits139U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mutex_waiter {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fred_ss {
    __slate_bits_0: __slate_bits::__SlateBits140U0,
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
__verify_pcpu_ptr__3!();
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

#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x0_0 {
    __slate_bits_0: __slate_bits::__SlateBits144U0,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct leaf_0x1_0 {
    __slate_bits_0: __slate_bits::__SlateBits145U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct leaf_parse_info {
    nr_entries: u32,
}


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
#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_leaves {
    leaf_0x0_0: [leaf_0x0_0; 1],
    leaf_0x0_0_info: leaf_parse_info,
    leaf_0x1_0: [leaf_0x1_0; 1],
    leaf_0x1_0_info: leaf_parse_info,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuid_table {
    leaves: cpuid_leaves,
}


#[inline(always)]
fn __phys_addr_nodebug(mut x: u64) -> u64 {
let y: u64 = x.wrapping_sub(_AC__2!());
// use the carry flag to determine if x was < __START_KERNEL_map
x = y.wrapping_add(if x > y { unsafe { phys_base } } else { _AC__2!().wrapping_sub(unsafe { page_offset_base }) });
x
}

#[repr(C)]
#[derive(Clone, Copy)]
struct range {
    start: u64,
    end: u64,
}


#[inline(always)]
fn native_pause() {
unsafe {
core::arch::asm!("pause", options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
struct cpuinfo_topology {
    apicid: u32,
    initial_apicid: u32,
    pkg_id: u32,
    die_id: u32,
    cu_id: u32,
    core_id: u32,
    logical_pkg_id: u32,
    logical_die_id: u32,
    logical_core_id: u32,
    amd_node_id: u32,
    llc_id: u32,
    l2c_id: u32,
    __slate_anon_12: __SlateRecord_processor_h_111,
    cpu_type: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct cpuinfo_x86 {
    __slate_anon_0: __SlateRecord_processor_h_136,
    x86_stepping: u8,
    __slate_anon_2: __SlateRecord_processor_h_154,
    x86_tlbsize: i32,
    vmx_capability: [u32; 5],
    x86_virt_bits: u8,
    x86_phys_bits: u8,
    extended_cpuid_level: u32,
    cpuid_level: i32,
    __slate_anon_9: __SlateRecord_processor_h_177,
    x86_vendor_id: [u8; 16],
    x86_model_id: [u8; 64],
    topo: cpuinfo_topology,
    cpuid: cpuid_table,
    x86_cache_size: u32,
    x86_cache_alignment: i32,
    x86_cache_max_rmid: i32,
    x86_cache_occ_scale: i32,
    x86_cache_mbm_width_offset: i32,
    x86_power: i32,
    loops_per_jiffy: u64,
    ppin: u64,
    x86_clflush_size: u16,
    booted_cores: u16,
    cpu_index: u16,
    smt_active: bool,
    microcode: u32,
    x86_cache_bits: u8,
    __slate_bits_0: __slate_bits::__SlateBits143U0,
}


// sizeof(unsigned long) coming from an extra "long" at the end of the
// iobitmap. The limit is inclusive, i.e. the last valid byte.
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
struct futex_pi_state {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct robust_list_head {
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
struct futex_private_hash {
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
struct pid_namespace {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct sem_undo_list {
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


// uses I/O bitmap
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
struct tracepoint_func {
    func: *mut (),
    data: *mut (),
    prio: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct hrtimer_clock_base {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct tracepoint_ext {
    regfunc: Option<unsafe extern "C" fn() -> i32>,
    unregfunc: Option<unsafe extern "C" fn()>,
    __slate_bits_0: __slate_bits::__SlateBits184U0,
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
struct seccomp_filter {
}


// Always evaluate the 'subclass' argument to avoid that the compiler
// warns about set-but-not-used variables when building with
// CONFIG_DEBUG_LOCK_ALLOC=n and with W=1.
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_spinlock_h_558 {
    lock: *mut raw_spinlock,
}


#[inline(always)]
fn class_raw_spinlock_destructor(_T: *mut __SlateRecord_spinlock_h_558) {
raw_spin_unlock!(unsafe { (*_T).lock });
}

#[inline(always)]
fn class_raw_spinlock_constructor(l: *mut raw_spinlock) -> __SlateRecord_spinlock_h_558 {
let mut _t: __SlateRecord_spinlock_h_558 = __SlateRecord_spinlock_h_558 { lock: l };
let _T: *mut __SlateRecord_spinlock_h_558 = core::ptr::addr_of_mut!(_t);
raw_spin_lock!(unsafe { (*_T).lock });
return _t;
}

#[inline(always)]
fn __class_raw_spinlock_cleanup_ctx(_T: *mut *mut __SlateRecord_spinlock_h_558) {
}

static class_raw_spinlock_irqsave_is_conditional: bool = 0i32 != 0i32;

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
struct __SlateRecord_spinlock_h_594 {
    lock: *mut raw_spinlock,
    flags: u64,
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
struct __SlateRecord_mutex_h_253 {
    lock: *mut mutex,
}


#[inline(always)]
fn class_mutex_destructor(_T: *mut __SlateRecord_mutex_h_253) {
unsafe { mutex_unlock((*_T).lock) };
}

#[repr(C)]
#[derive(Clone, Copy)]
struct workqueue_struct {
}


#[inline(always)]
fn class_mutex_constructor(l: *mut mutex) -> __SlateRecord_mutex_h_253 {
let mut _t: __SlateRecord_mutex_h_253 = __SlateRecord_mutex_h_253 { lock: l };
let _T: *mut __SlateRecord_mutex_h_253 = core::ptr::addr_of_mut!(_t);
unsafe { mutex_lock((*_T).lock) };
return _t;
}

#[repr(C)]
#[derive(Clone, Copy)]
struct rseq {
}


#[inline(always)]
fn __class_mutex_cleanup_ctx(_T: *mut *mut __SlateRecord_mutex_h_253) {
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
struct seqcount {
    sequence: u32,
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
struct rb_node {
    __rb_parent_color: u64,
    rb_right: *mut rb_node,
    rb_left: *mut rb_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_unwind_deferred_types_h_28 {
    cpu: u32,
    cnt: u32,
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
struct rb_root {
    rb_node: *mut rb_node,
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
struct rb_root_cached {
    rb_root: rb_root,
    rb_leftmost: *mut rb_node,
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
struct rq_flags {
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
struct __SlateRecord_sched_h_791 {
    blocked: u8,
    need_qs: u8,
    exp_hint: u8,
    need_mb: u8,
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
struct rlimit {
    rlim_cur: u64,
    rlim_max: u64,
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
struct sigpending {
    list: list_head,
    signal: __SlateRecord_signal_h_25,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct gendisk {
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
struct vm_struct {
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
union __SlateRecord_maple_tree_h_261 {
    ma_lock: spinlock,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct posix_cputimer_base {
    nextevt: u64,
    tqhead: timerqueue_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rwsem_waiter {
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
struct uprobe {
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
    __slate_bits_0: __slate_bits::__SlateBits319U0,
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


// Number of contexts where an event can trigger:
//      task, softirq, hardirq, nmi.
#[repr(C)]
#[derive(Clone, Copy)]
struct wake_q_node {
    next: *mut wake_q_node,
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
    __slate_bits_0: __slate_bits::__SlateBits69U0,
    __slate_pad_65: [u8; 3],
    __slate_bits_1: __slate_bits::__SlateBits69U1,
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
    __slate_bits_2: __slate_bits::__SlateBits69U2,
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
struct rw_semaphore {
    count: __SlateRecord_types_h_197,
    owner: __SlateRecord_types_h_197,
    osq: optimistic_spin_queue,
    wait_lock: raw_spinlock,
    first_waiter: *mut rwsem_waiter,
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
struct ghcb {
}


fn native_read_msr_safe(mut msr: u32, p: &mut u64) -> i32 {
let mut err: i32 = 0;
let mut low: u64 = 0;
let mut high: u64 = 0;
unsafe {
let __t0: u32;
let __t1: u64;
let __t2: u64;
core::arch::asm!("1: rdmsr ; xor {0:e},{0:e}\n2:\n\t .pushsection __ex_table, \"aM\", @progbits, 12\n .balign 4\n .long (1b) - .\n .long (2b) - .\n.macro extable_type_reg type:req reg:req\n.set .Lfound, 0\n.set .Lregnr, 0\n.irp rs,rax,rcx,rdx,rbx,rsp,rbp,rsi,rdi,r8,r9,r10,r11,r12,r13,r14,r15\n.ifc \\reg, %\\rs\n.set .Lfound, .Lfound+1\n.long \\type + (.Lregnr << 8)\n.endif\n.set .Lregnr, .Lregnr+1\n.endr\n.set .Lregnr, 0\n.irp rs,eax,ecx,edx,ebx,esp,ebp,esi,edi,r8d,r9d,r10d,r11d,r12d,r13d,r14d,r15d\n.ifc \\reg, %\\rs\n.set .Lfound, .Lfound+1\n.long \\type + (.Lregnr << 8)\n.endif\n.set .Lregnr, .Lregnr+1\n.endr\n.if (.Lfound != 1)\n.error \"extable_type_reg: bad register argument\"\n.endif\n.endm\nextable_type_reg reg={0:e}, type=11 \n.purgem extable_type_reg\n .popsection\n", lateout(reg) __t0, lateout("ax") __t1, lateout("dx") __t2, in("cx") msr as u32, options(nostack, att_syntax));
err = unsafe { core::mem::transmute::<u32, i32>(__t0 as u32) };
low = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
high = unsafe { core::mem::transmute::<u64, u64>(__t2 as u64) };
}
if static_branch_unlikely!(unsafe { core::ptr::addr_of_mut!(__tracepoint_read_msr.key) }) != 0 {
unsafe { do_trace_read_msr(msr, EAX_EDX_VAL!(low, high), err) };
}
unsafe {
*(p as *mut u64) = EAX_EDX_VAL!(low, high);
}
err
}

fn rdmsrq_safe(msr: u32, p: &mut u64) -> i32 {
native_read_msr_safe(msr, p)
}

// Instruction opcode for WRMSRNS supported in binutils >= 2.40
#[inline(always)]
fn rdtsc() -> u64 {
let mut low: u64 = 0;
let mut high: u64 = 0;
unsafe {
let __t0: u64;
let __t1: u64;
core::arch::asm!("rdtsc", lateout("ax") __t0, lateout("dx") __t1, options(nostack, att_syntax));
low = unsafe { core::mem::transmute::<u64, u64>(__t0 as u64) };
high = unsafe { core::mem::transmute::<u64, u64>(__t1 as u64) };
}
EAX_EDX_VAL!(low, high)
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
    __slate_bits_0: __slate_bits::__SlateBits440U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct user_namespace {
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
struct mnt_idmap {
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
struct assoc_array_ptr {
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
struct key_type {
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
struct mpc_ioapic {
    r#type: u8,
    apicid: u8,
    apicver: u8,
    flags: u8,
    apicaddr: u32,
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
struct mpc_intsrc {
    r#type: u8,
    irqtype: u8,
    irqflag: u16,
    srcbus: u8,
    srcbusirq: u8,
    dstapic: u8,
    dstirq: u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_mpparse {
    setup_ioapic_ids: Option<unsafe extern "C" fn()>,
    find_mptable: Option<unsafe extern "C" fn()>,
    early_parse_smp_cfg: Option<unsafe extern "C" fn()>,
    parse_smp_cfg: Option<unsafe extern "C" fn()>,
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
struct x86_init_resources {
    probe_roms: Option<unsafe extern "C" fn()>,
    reserve_resources: Option<unsafe extern "C" fn()>,
    memory_setup: Option<unsafe extern "C" fn() -> *mut u8>,
    dmi_setup: Option<unsafe extern "C" fn()>,
    realmode_limit: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_cred_h_147 {
    non_rcu: i32,
    rcu: callback_head,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_irqs {
    pre_vector_init: Option<unsafe extern "C" fn()>,
    intr_init: Option<unsafe extern "C" fn()>,
    intr_mode_select: Option<unsafe extern "C" fn()>,
    intr_mode_init: Option<unsafe extern "C" fn()>,
    create_pci_msi_domain: Option<unsafe extern "C" fn() -> *mut irq_domain>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pidfs_attr {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_oem {
    arch_setup: Option<unsafe extern "C" fn()>,
    banner: Option<unsafe extern "C" fn()>,
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
struct clock_event_device {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_paging {
    pagetable_init: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_timers {
    setup_percpu_clockev: Option<unsafe extern "C" fn()>,
    timer_init: Option<unsafe extern "C" fn()>,
    wallclock_init: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_iommu {
    iommu_init: Option<unsafe extern "C" fn() -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_pci {
    arch_init: Option<unsafe extern "C" fn() -> i32>,
    init: Option<unsafe extern "C" fn() -> i32>,
    init_irq: Option<unsafe extern "C" fn()>,
    fixup_irqs: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_hyper_init {
    init_platform: Option<unsafe extern "C" fn()>,
    guest_late_init: Option<unsafe extern "C" fn()>,
    x2apic_available: Option<unsafe extern "C" fn() -> bool>,
    msi_ext_dest_id: Option<unsafe extern "C" fn() -> bool>,
    init_mem_mapping: Option<unsafe extern "C" fn()>,
    init_after_bootmem: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_acpi {
    set_root_pointer: Option<unsafe extern "C" fn(u64)>,
    get_root_pointer: Option<unsafe extern "C" fn() -> u64>,
    reduced_hw_early_init: Option<unsafe extern "C" fn()>,
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
struct x86_guest {
    enc_status_change_prepare: Option<unsafe extern "C" fn(u64, i32, bool) -> i32>,
    enc_status_change_finish: Option<unsafe extern "C" fn(u64, i32, bool) -> i32>,
    enc_tlb_flush_required: Option<unsafe extern "C" fn(bool) -> bool>,
    enc_cache_flush_required: Option<unsafe extern "C" fn() -> bool>,
    enc_kexec_begin: Option<unsafe extern "C" fn()>,
    enc_kexec_finish: Option<unsafe extern "C" fn()>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_init_ops {
    resources: x86_init_resources,
    mpparse: x86_init_mpparse,
    irqs: x86_init_irqs,
    oem: x86_init_oem,
    paging: x86_init_paging,
    timers: x86_init_timers,
    iommu: x86_init_iommu,
    pci: x86_init_pci,
    hyper: x86_hyper_init,
    acpi: x86_init_acpi,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_legacy_devices {
    pnpbios: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_legacy_features {
    i8042: u32,
    rtc: i32,
    warm_reset: i32,
    no_vga: i32,
    reserve_bios_regions: i32,
    devices: x86_legacy_devices,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_hyper_runtime {
    pin_vcpu: Option<unsafe extern "C" fn(i32)>,
    sev_es_hcall_prepare: Option<unsafe extern "C" fn(*mut ghcb, *mut pt_regs)>,
    sev_es_hcall_finish: Option<unsafe extern "C" fn(*mut ghcb, *mut pt_regs) -> bool>,
    is_private_mmio: Option<unsafe extern "C" fn(u64) -> bool>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_platform_ops {
    calibrate_cpu: Option<unsafe extern "C" fn() -> u64>,
    calibrate_tsc: Option<unsafe extern "C" fn() -> u64>,
    get_wallclock: Option<unsafe extern "C" fn(*mut timespec64)>,
    set_wallclock: Option<unsafe extern "C" fn(*const timespec64) -> i32>,
    iommu_shutdown: Option<unsafe extern "C" fn()>,
    is_untracked_pat_range: Option<unsafe extern "C" fn(u64, u64) -> bool>,
    nmi_init: Option<unsafe extern "C" fn()>,
    get_nmi_reason: Option<unsafe extern "C" fn() -> u8>,
    save_sched_clock_state: Option<unsafe extern "C" fn()>,
    restore_sched_clock_state: Option<unsafe extern "C" fn()>,
    apic_post_init: Option<unsafe extern "C" fn()>,
    legacy: x86_legacy_features,
    set_legacy_features: Option<unsafe extern "C" fn()>,
    realmode_reserve: Option<unsafe extern "C" fn()>,
    realmode_init: Option<unsafe extern "C" fn()>,
    hyper: x86_hyper_runtime,
    guest: x86_guest,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_apic_ops {
    io_apic_read: Option<unsafe extern "C" fn(u32, u32) -> u32>,
    restore: Option<unsafe extern "C" fn()>,
}


#[unsafe(no_mangle)]
static mut mp_bus_not_pci: __SlateAlign16<[u64; 4]> = __SlateAlign16(unsafe { core::mem::zeroed() });

fn cpu_to_node(cpu: i32) -> i32 {
per_cpu__2!(numa_node, cpu)
}

#[repr(C)]
#[derive(Clone, Copy)]
struct percpu_ref_data {
    count: __SlateRecord_types_h_197,
    release: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    confirm_switch: Option<unsafe extern "C" fn(*mut percpu_ref)>,
    __slate_bits_0: __slate_bits::__SlateBits494U0,
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
union __SlateRecord_quota_h_69 {
    uid: __SlateRecord_uidgid_types_h_9,
    gid: __SlateRecord_uidgid_types_h_13,
    projid: __SlateRecord_projid_h_24,
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


fn cc_mkenc(val: u64) -> u64 {
val
}

fn cc_mkdec(val: u64) -> u64 {
val
}

fn __set_fixmap(idx: u32, phys: u64, flags: pgprot) {
unsafe { native_set_fixmap(idx as u32, phys, flags) };
}

// FIXMAP_PAGE_NOCACHE is used for MMIO. Memory encryption is not
// supported for MMIO addresses, so make sure that the memory encryption
// mask is not part of the page attributes.
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
// Driver has marked this resource busy
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
struct backing_dev_info {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct block_device {
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
#[inline(always)]
fn _kmalloc_node_noprof(size: u64, flags: u32, node: i32, token: __SlateRecord_slab_h_532) -> *mut () {
if (unsafe { __slate_intrinsic_f217c4720f04f6db(size) }) && size != 0 {
let mut index: u32 = 0;
if size > 8192u64 {
return unsafe { __kmalloc_large_node_noprof(size, flags, node) };
}
index = kmalloc_index!(size);
return unsafe { __kmalloc_cache_node_noprof(kmalloc_caches[(kmalloc_type(flags, token) as u32) as usize][index as usize], flags, node, size) };
}
unsafe { __kmalloc_node_noprof(PASS_KMALLOC_PARAMS!(size), flags, node) }
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


fn _kzalloc_noprof(size: u64, flags: u32, token: __SlateRecord_slab_h_532) -> *mut () {
_kmalloc_noprof(size, flags | (BIT!(8i32) as u32), token)
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


#[repr(C)]
#[derive(Clone, Copy)]
struct exception_table_entry {
    insn: i32,
    fixup: i32,
    data: i32,
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
struct key_tag {
    rcu: callback_head,
    usage: refcount_struct,
    removed: bool,
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
struct keyring_index_key {
    hash: u64,
    __slate_anon_1: __SlateRecord_key_h_117,
    r#type: *mut key_type,
    domain_tag: *mut key_tag,
    description: *const u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fs_context {
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
struct io_comp_batch {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct fiemap_extent_info {
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
struct swap_info_struct {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct rhash_head {
    next: *mut rhash_head,
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
struct upid {
    nr: i32,
    ns: *mut pid_namespace,
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
struct posix_acl {
}


fn hrtimer_get_expires(timer: *const hrtimer) -> i64 {
unsafe { (*timer).node.expires }
}

fn hrtimer_get_softexpires(timer: *const hrtimer) -> i64 {
unsafe { (*timer)._softexpires }
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
struct io_uring_cmd {
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
    __slate_bits_0: __slate_bits::__SlateBits322U0,
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
if !(__native_word!() || ((const { core::mem::size_of::<u64>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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

#[repr(C)]
#[derive(Clone, Copy)]
struct sock {
}


fn rlimit(limit: u32) -> u64 {
task_rlimit(get_current() as *const task_struct, limit)
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
union __SlateRecord_interrupt_h_125 {
    dev_id: *mut (),
    percpu_dev_id: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct proc_dir_entry {
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
union __SlateRecord_hw_irq_h_82 {
    ioapic: ioapic_alloc_info,
    uv: uv_alloc_info,
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
struct wake_irq {
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
struct dev_pm_qos {
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
if !(__native_word__2!() || ((const { core::mem::size_of::<*mut mnt_idmap>() }) as u64) == ((const { core::mem::size_of::<i64>() }) as u64)) {
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
union __SlateRecord_moduleparam_h_89 {
    arg: *mut (),
    str: *const kparam_string,
    arr: *const kparam_array,
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
struct module_param_attrs {
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
struct driver_private {
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_driver_h_126 {
    post_unbind_rust: Option<unsafe extern "C" fn(*mut device)>,
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
    __slate_bits_0: __slate_bits::__SlateBits681U0,
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
    __slate_bits_0: __slate_bits::__SlateBits633U0,
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


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irqaction {
    handler: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    __slate_anon_1: __SlateRecord_interrupt_h_125,
    affinity: *const cpumask,
    next: *mut irqaction,
    thread_fn: Option<unsafe extern "C" fn(i32, *mut ()) -> u32>,
    thread: *mut task_struct,
    secondary: *mut irqaction,
    irq: u32,
    flags: u32,
    thread_flags: u64,
    thread_mask: u64,
    name: *const u8,
    dir: *mut proc_dir_entry,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_notify {
    irq: u32,
    kref: kref,
    work: work_struct,
    notify: Option<unsafe extern "C" fn(*mut irq_affinity_notify, *const cpumask)>,
    release: Option<unsafe extern "C" fn(*mut kref)>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_affinity_desc {
    mask: cpumask,
    __slate_bits_0: __slate_bits::__SlateBits702U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_26 {
    __slate_anon_0: __SlateRecord_msi_h_26_2,
    dmar_subhandle: u32,
}


#[repr(C, align(4))]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_26_2 {
    __slate_bits_0: __slate_bits::__SlateBits839U0,
}


fn readl(mut addr: *const ()) -> u32 {
let mut ret: u32 = 0;
unsafe {
let __t0: u32;
core::arch::asm!("movl ({1:r}),{0:e}", lateout(reg) __t0, in(reg) addr as *mut u32, options(nostack, att_syntax));
ret = unsafe { core::mem::transmute::<u32, u32>(__t0 as u32) };
}
ret
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_49 {
    __slate_anon_0: __SlateRecord_msi_h_49_2,
    __slate_anon_1: __SlateRecord_msi_h_49_3,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_2 {
    __slate_bits_0: __slate_bits::__SlateBits843U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_49_3 {
    __slate_bits_0: __slate_bits::__SlateBits844U0,
}


fn writel(mut val: u32, mut addr: *mut ()) {
unsafe {
core::arch::asm!("movl {0:e},({1:r})", in(reg) val as u32, in(reg) addr as *mut u32, options(nostack, att_syntax));
}
}

#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_62 {
    address_lo: u32,
    arch_addr_lo: x86_msi_addr_lo,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_66 {
    address_hi: u32,
    arch_addr_hi: x86_msi_addr_hi,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_70 {
    data: u32,
    arch_data: x86_msi_data,
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
union __SlateRecord_msi_h_109 {
    msi_mask: u32,
    msix_ctrl: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_msi_h_113 {
    __slate_bits_0: __slate_bits::__SlateBits853U0,
    default_irq: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_msi_h_122 {
    mask_pos: u8,
    mask_base: *mut (),
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
union __SlateRecord_msi_h_202 {
    pci: pci_msi_desc,
    data: msi_desc_data,
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
struct irq_common_data {
    state_use_accessors: u32,
    node: u32,
    handler_data: *mut (),
    msi_desc: *mut msi_desc,
    affinity: [cpumask; 1],
    effective_affinity: [cpumask; 1],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_data {
    mask: u32,
    irq: u32,
    hwirq: u64,
    common: *mut irq_common_data,
    chip: *mut irq_chip,
    domain: *mut irq_domain,
    parent_data: *mut irq_data,
    chip_data: *mut (),
}


fn irqd_is_setaffinity_pending(d: *mut irq_data) -> bool {
__irqd_to_state!(d) & 256 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_io_apic_h_21 {
    __slate_bits_0: __slate_bits::__SlateBits863U0,
}


fn irqd_irq_masked(d: *mut irq_data) -> bool {
__irqd_to_state!(d) & 131072 != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_io_apic_h_32 {
    __slate_bits_0: __slate_bits::__SlateBits865U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip {
    name: *const u8,
    irq_startup: Option<unsafe extern "C" fn(*mut irq_data) -> u32>,
    irq_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_enable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_disable: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_mask_ack: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_unmask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_eoi: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_set_affinity: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask, bool) -> i32>,
    irq_pre_redirect: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_retrigger: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_set_type: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_set_wake: Option<unsafe extern "C" fn(*mut irq_data, u32) -> i32>,
    irq_bus_lock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_bus_sync_unlock: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_suspend: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_resume: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_pm_shutdown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_calc_mask: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_print_chip: Option<unsafe extern "C" fn(*mut irq_data, *mut seq_file)>,
    irq_request_resources: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_release_resources: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_compose_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_write_msi_msg: Option<unsafe extern "C" fn(*mut irq_data, *mut msi_msg)>,
    irq_get_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, *mut bool) -> i32>,
    irq_set_irqchip_state: Option<unsafe extern "C" fn(*mut irq_data, u32, bool) -> i32>,
    irq_set_vcpu_affinity: Option<unsafe extern "C" fn(*mut irq_data, *mut ()) -> i32>,
    ipi_send_single: Option<unsafe extern "C" fn(*mut irq_data, u32)>,
    ipi_send_mask: Option<unsafe extern "C" fn(*mut irq_data, *const cpumask)>,
    irq_nmi_setup: Option<unsafe extern "C" fn(*mut irq_data) -> i32>,
    irq_nmi_teardown: Option<unsafe extern "C" fn(*mut irq_data)>,
    irq_force_complete_move: Option<unsafe extern "C" fn(*mut irq_data)>,
    flags: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_io_apic_h_43 {
    __slate_bits_0: __slate_bits::__SlateBits867U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irqstat {
    cnt: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_io_apic_h_52 {
    __slate_bits_0: __slate_bits::__SlateBits869U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_redirect {
    work: irq_work,
    target_cpu: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union __SlateRecord_io_apic_h_59 {
    __slate_anon_0: __SlateRecord_io_apic_h_60,
    __slate_anon_1: __SlateRecord_io_apic_h_74,
    __slate_anon_2: __SlateRecord_io_apic_h_83,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_io_apic_h_60 {
    __slate_bits_0: __slate_bits::__SlateBits872U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_io_apic_h_74 {
    __slate_bits_0: __slate_bits::__SlateBits873U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_io_apic_h_83 {
    __slate_bits_0: __slate_bits::__SlateBits874U0,
}


#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct irq_desc {
    irq_common_data: irq_common_data,
    irq_data: irq_data,
    kstat_irqs: *mut irqstat,
    handle_irq: Option<unsafe extern "C" fn(*mut irq_desc)>,
    action: *mut irqaction,
    status_use_accessors: u32,
    core_internal_state__do_not_mess_with_it: u32,
    depth: u32,
    wake_depth: u32,
    tot_count: u64,
    last_unhandled: u64,
    irq_count: u32,
    irqs_unhandled: u32,
    threads_handled: __SlateRecord_types_h_190,
    threads_handled_last: i32,
    lock: raw_spinlock,
    percpu_enabled: *mut cpumask,
    redirect: irq_redirect,
    affinity_hint: *const cpumask,
    affinity_notify: *mut irq_affinity_notify,
    pending_mask: [cpumask; 1],
    threads_oneshot: u64,
    threads_active: __SlateRecord_types_h_190,
    wait_for_threads: wait_queue_head,
    nr_actions: u32,
    no_suspend_depth: u32,
    cond_suspend_depth: u32,
    force_resume_depth: u32,
    dir: *mut proc_dir_entry,
    refcnt: __SlateRecord_types_h_202,
    rcu: callback_head,
    kobj: kobject,
    request_mutex: mutex,
    parent_irq: i32,
    owner: *mut module,
    name: *const u8,
    resend_node: hlist_node,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ioapic_alloc_info {
    pin: i32,
    node: i32,
    __slate_bits_0: __slate_bits::__SlateBits722U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct uv_alloc_info {
    limit: i32,
    blade: i32,
    offset: u64,
    name: *mut u8,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_alloc_info {
    r#type: u32,
    flags: u32,
    devid: u32,
    hwirq: u64,
    mask: *const cpumask,
    desc: *mut msi_desc,
    data: *mut (),
    __slate_anon_7: __SlateRecord_hw_irq_h_82,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_cfg {
    dest_apicid: u32,
    vector: u32,
}


fn irq_set_status_flags(irq: u32, set: u64) {
unsafe { irq_modify_status(irq, 0u64, set) };
}

fn irq_clear_status_flags(irq: u32, clr: u64) {
unsafe { irq_modify_status(irq, clr, 0u64) };
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
    pub struct __SlateBits319U0 {
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
    pub struct __SlateBits69U0 {
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
    pub struct __SlateBits69U1 {
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
    pub struct __SlateBits322U0 {
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
    pub struct __SlateBits69U2 {
        #[bits(1)]
        pub mce_ripv: u64,
        #[bits(1)]
        pub mce_whole_page: u64,
        #[bits(62)]
        pub __mce_reserved: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits107U0 {
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
    pub struct __SlateBits440U0 {
        #[bits(1)]
        pub hide_from_rmap_until_complete: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits494U0 {
        #[bits(1)]
        pub force_atomic: bool,
        #[bits(1)]
        pub allow_reinit: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits681U0 {
        #[bits(1)]
        pub mmapped: bool,
        #[bits(1)]
        pub released: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits633U0 {
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
    pub struct __SlateBits106U0 {
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
    pub struct __SlateBits153U0 {
        #[bits(24)]
        pub intel_native_model_id: u32,
        #[bits(8)]
        pub intel_type: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits154U0 {
        #[bits(16)]
        pub amd_num_processors: u32,
        #[bits(8)]
        pub amd_power_eff_ranking: u32,
        #[bits(4)]
        pub amd_native_model_id: u32,
        #[bits(4)]
        pub amd_type: u32,
    }
    #[bitfields::bitfield([u8; 16], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits144U0 {
        #[bits(32)]
        pub max_std_leaf: u32,
        #[bits(32)]
        pub cpu_vendorid_0: u32,
        #[bits(32)]
        pub cpu_vendorid_2: u32,
        #[bits(32)]
        pub cpu_vendorid_1: u32,
    }
    #[bitfields::bitfield([u8; 16], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits145U0 {
        #[bits(4)]
        pub stepping: u32,
        #[bits(4)]
        pub base_model: u32,
        #[bits(4)]
        pub base_family_id: u32,
        #[bits(2)]
        pub cpu_type: u32,
        #[bits(2, access = na)]
        pub __slate_pad_4: u8,
        #[bits(4)]
        pub ext_model: u32,
        #[bits(8)]
        pub ext_family: u32,
        #[bits(4, access = na)]
        pub __slate_pad_7: u8,
        #[bits(8)]
        pub brand_id: u32,
        #[bits(8)]
        pub clflush_size: u32,
        #[bits(8)]
        pub n_logical_cpu: u32,
        #[bits(8)]
        pub local_apic_id: u32,
        #[bits(1)]
        pub sse3: u32,
        #[bits(1)]
        pub pclmulqdq: u32,
        #[bits(1)]
        pub dtes64: u32,
        #[bits(1)]
        pub monitor: u32,
        #[bits(1)]
        pub dscpl: u32,
        #[bits(1)]
        pub vmx: u32,
        #[bits(1)]
        pub smx: u32,
        #[bits(1)]
        pub est: u32,
        #[bits(1)]
        pub tm2: u32,
        #[bits(1)]
        pub ssse3: u32,
        #[bits(1)]
        pub cntxt_id: u32,
        #[bits(1)]
        pub sdbg: u32,
        #[bits(1)]
        pub fma: u32,
        #[bits(1)]
        pub cx16: u32,
        #[bits(1)]
        pub xtpr_update: u32,
        #[bits(1)]
        pub pdcm: u32,
        #[bits(1, access = na)]
        pub __slate_pad_28: u8,
        #[bits(1)]
        pub pcid: u32,
        #[bits(1)]
        pub dca: u32,
        #[bits(1)]
        pub sse4_1: u32,
        #[bits(1)]
        pub sse4_2: u32,
        #[bits(1)]
        pub x2apic: u32,
        #[bits(1)]
        pub movbe: u32,
        #[bits(1)]
        pub popcnt: u32,
        #[bits(1)]
        pub tsc_deadline_timer: u32,
        #[bits(1)]
        pub aes: u32,
        #[bits(1)]
        pub xsave: u32,
        #[bits(1)]
        pub osxsave: u32,
        #[bits(1)]
        pub avx: u32,
        #[bits(1)]
        pub f16c: u32,
        #[bits(1)]
        pub rdrand: u32,
        #[bits(1)]
        pub guest_status: u32,
        #[bits(1)]
        pub fpu: u32,
        #[bits(1)]
        pub vme: u32,
        #[bits(1)]
        pub de: u32,
        #[bits(1)]
        pub pse: u32,
        #[bits(1)]
        pub tsc: u32,
        #[bits(1)]
        pub msr: u32,
        #[bits(1)]
        pub pae: u32,
        #[bits(1)]
        pub mce: u32,
        #[bits(1)]
        pub cx8: u32,
        #[bits(1)]
        pub apic: u32,
        #[bits(1, access = na)]
        pub __slate_pad_54: u8,
        #[bits(1)]
        pub sep: u32,
        #[bits(1)]
        pub mtrr: u32,
        #[bits(1)]
        pub pge: u32,
        #[bits(1)]
        pub mca: u32,
        #[bits(1)]
        pub cmov: u32,
        #[bits(1)]
        pub pat: u32,
        #[bits(1)]
        pub pse36: u32,
        #[bits(1)]
        pub psn: u32,
        #[bits(1)]
        pub clflush: u32,
        #[bits(1, access = na)]
        pub __slate_pad_64: u8,
        #[bits(1)]
        pub ds: u32,
        #[bits(1)]
        pub acpi: u32,
        #[bits(1)]
        pub mmx: u32,
        #[bits(1)]
        pub fxsr: u32,
        #[bits(1)]
        pub sse: u32,
        #[bits(1)]
        pub sse2: u32,
        #[bits(1)]
        pub selfsnoop: u32,
        #[bits(1)]
        pub htt: u32,
        #[bits(1)]
        pub tm: u32,
        #[bits(1)]
        pub ia64: u32,
        #[bits(1)]
        pub pbe: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits143U0 {
        #[bits(1)]
        pub initialized: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits184U0 {
        #[bits(1)]
        pub faultable: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits745U0 {
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
    pub struct __SlateBits742U0 {
        #[bits(1)]
        pub active: bool,
        #[bits(1)]
        pub autosleep_enabled: bool,
        #[bits(6, access = na)]
        pub __slate_pad_2: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits745U1 {
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
    pub struct __SlateBits745U2 {
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
    pub struct __SlateBits745U3 {
        #[bits(1)]
        pub detach_power_off: bool,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits843U0 {
        #[bits(2)]
        pub reserved_0: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(1)]
        pub redirect_hint: u32,
        #[bits(1)]
        pub reserved_1: u32,
        #[bits(7)]
        pub virt_destid_8_14: u32,
        #[bits(8)]
        pub destid_0_7: u32,
        #[bits(12)]
        pub base_address: u32,
    }
    #[bitfields::bitfield(u32, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits844U0 {
        #[bits(2)]
        pub dmar_reserved_0: u32,
        #[bits(1)]
        pub dmar_index_15: u32,
        #[bits(1)]
        pub dmar_subhandle_valid: u32,
        #[bits(1)]
        pub dmar_format: u32,
        #[bits(15)]
        pub dmar_index_0_14: u32,
        #[bits(12)]
        pub dmar_base_address: u32,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits846U0 {
        #[bits(8)]
        pub reserved: u32,
        #[bits(24)]
        pub destid_8_31: u32,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits839U0 {
        #[bits(8)]
        pub vector: u32,
        #[bits(3)]
        pub delivery_mode: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(2)]
        pub reserved: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub is_level: u32,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits702U0 {
        #[bits(1)]
        pub is_managed: u32,
        #[bits(7, access = na)]
        pub __slate_pad_1: u8,
    }
    #[bitfields::bitfield(u16, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits853U0 {
        #[bits(1)]
        pub is_msix: u8,
        #[bits(3)]
        pub multiple: u8,
        #[bits(3)]
        pub multi_cap: u8,
        #[bits(1)]
        pub can_mask: u8,
        #[bits(1)]
        pub is_64: u8,
        #[bits(1)]
        pub is_virtual: u8,
        #[bits(6, access = na)]
        pub __slate_pad_6: u8,
    }
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits722U0 {
        #[bits(1)]
        pub is_level: u32,
        #[bits(1)]
        pub active_low: u32,
        #[bits(1)]
        pub valid: u32,
        #[bits(5, access = na)]
        pub __slate_pad_3: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits139U0 {
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
    pub struct __SlateBits140U0 {
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
    #[bitfields::bitfield(u8, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits875U0 {
        #[bits(1)]
        pub disable_esr: u32,
        #[bits(1)]
        pub dest_mode_logical: u32,
        #[bits(1)]
        pub x2apic_set_max_apicid: u32,
        #[bits(1)]
        pub nmi_to_offline_cpu: u32,
        #[bits(4, access = na)]
        pub __slate_pad_4: u8,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits872U0 {
        #[bits(8)]
        pub vector: u64,
        #[bits(3)]
        pub delivery_mode: u64,
        #[bits(1)]
        pub dest_mode_logical: u64,
        #[bits(1)]
        pub delivery_status: u64,
        #[bits(1)]
        pub active_low: u64,
        #[bits(1)]
        pub irr: u64,
        #[bits(1)]
        pub is_level: u64,
        #[bits(1)]
        pub masked: u64,
        #[bits(15)]
        pub reserved_0: u64,
        #[bits(17)]
        pub reserved_1: u64,
        #[bits(7)]
        pub virt_destid_8_14: u64,
        #[bits(8)]
        pub destid_0_7: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits873U0 {
        #[bits(8)]
        pub ir_shared_0: u64,
        #[bits(3)]
        pub ir_zero: u64,
        #[bits(1)]
        pub ir_index_15: u64,
        #[bits(5)]
        pub ir_shared_1: u64,
        #[bits(31)]
        pub ir_reserved_0: u64,
        #[bits(1)]
        pub ir_format: u64,
        #[bits(15)]
        pub ir_index_0_14: u64,
    }
    #[bitfields::bitfield(u64, c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits874U0 {
        #[bits(32)]
        pub w1: u64,
        #[bits(32)]
        pub w2: u64,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits869U0 {
        #[bits(1)]
        pub boot_DT: u32,
        #[bits(31)]
        pub __reserved_1: u32,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits867U0 {
        #[bits(24)]
        pub __reserved_2: u32,
        #[bits(4)]
        pub arbitration: u32,
        #[bits(4)]
        pub __reserved_1: u32,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits865U0 {
        #[bits(8)]
        pub version: u32,
        #[bits(7)]
        pub __reserved_2: u32,
        #[bits(1)]
        pub PRQ: u32,
        #[bits(8)]
        pub entries: u32,
        #[bits(8)]
        pub __reserved_1: u32,
    }
    #[bitfields::bitfield([u8; 4], c_names = true, new = false, from_into_bits = false, from_traits = false, default = false, debug = false, builder = false, bit_ops = false)]
    pub struct __SlateBits863U0 {
        #[bits(14)]
        pub __reserved_2: u32,
        #[bits(1)]
        pub LTS: u32,
        #[bits(1)]
        pub delivery_type: u32,
        #[bits(8)]
        pub __reserved_1: u32,
        #[bits(8)]
        pub ID: u32,
    }
}

fn irq_get_chip(irq: u32) -> *mut irq_chip {
let d: *mut irq_data = unsafe { irq_get_irq_data(irq) };
if !d.is_null() { unsafe { (*d).chip } } else { core::ptr::null_mut::<irq_chip>() }
}

fn irq_get_chip_data(irq: u32) -> *mut () {
let d: *mut irq_data = unsafe { irq_get_irq_data(irq) };
if !d.is_null() { unsafe { (*d).chip_data } } else { core::ptr::null_mut::<()>() }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_regs {
    enable: u64,
    disable: u64,
    mask: u64,
    ack: u64,
    eoi: u64,
    r#type: u64,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_type {
    chip: irq_chip,
    regs: irq_chip_regs,
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    r#type: u32,
    mask_cache_priv: u32,
    mask_cache: *mut u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_chip_generic {
    lock: raw_spinlock,
    reg_base: *mut (),
    reg_readl: Option<unsafe extern "C" fn(*mut ()) -> u32>,
    reg_writel: Option<unsafe extern "C" fn(u32, *mut ())>,
    suspend: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    resume: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    irq_base: u32,
    irq_cnt: u32,
    mask_cache: u32,
    wake_enabled: u32,
    wake_active: u32,
    num_ct: u32,
    private: *mut (),
    installed: u64,
    unused: u64,
    domain: *mut irq_domain,
    list: list_head,
    chip_types: [irq_chip_type; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic {
    irqs_per_chip: u32,
    num_chips: u32,
    irq_flags_to_clear: u32,
    irq_flags_to_set: u32,
    gc_flags: u32,
    exit: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
    gc: [*mut irq_chip_generic; 0],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_chip_generic_info {
    name: *const u8,
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    irqs_per_chip: u32,
    num_ct: u32,
    irq_flags_to_clear: u32,
    irq_flags_to_set: u32,
    gc_flags: u32,
    init: Option<unsafe extern "C" fn(*mut irq_chip_generic) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut irq_chip_generic)>,
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
    __slate_bits_0: __slate_bits::__SlateBits745U0,
    driver_flags: u32,
    lock: spinlock,
    entry: list_head,
    completion: completion,
    wakeup: *mut wakeup_source,
    work_in_progress: bool,
    __slate_bits_1: __slate_bits::__SlateBits745U1,
    suspend_timer: hrtimer,
    timer_expires: u64,
    work: work_struct,
    wait_queue: wait_queue_head,
    wakeirq: *mut wake_irq,
    usage_count: __SlateRecord_types_h_190,
    child_count: __SlateRecord_types_h_190,
    __slate_bits_2: __slate_bits::__SlateBits745U2,
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
    __slate_bits_3: __slate_bits::__SlateBits745U3,
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


core::arch::global_asm!(".section \".export_symbol\",\"a\" ; __export_symbol_IO_APIC_get_PCI_irq_vector: ; .asciz \"\" ; .ascii \"\" \"\\0\" ; .balign 8 ; .quad {0} ; .previous\n.section\t\".initcall6.init\", \"a\"\t\t\n__initcall__kmod_io_apic__662_2340_ioapic_init_ops6:\t\t\t\n.long\t{1} - .\t\n.previous\t\t\t\t\t\n", sym IO_APIC_get_PCI_irq_vector, sym ioapic_init_ops, options(att_syntax));

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
    __slate_bits_0: __slate_bits::__SlateBits742U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
union msi_instance_cookie {
    value: u64,
    ptr: *mut (),
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
struct cdev {
    kobj: kobject,
    owner: *mut module,
    ops: *const file_operations,
    list: list_head,
    dev: u32,
    count: u32,
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
struct poll_table_struct {
    _qproc: Option<unsafe extern "C" fn(*mut file, *mut wait_queue_head, *mut poll_table_struct)>,
    _key: u32,
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


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec {
    fwnode: *mut fwnode_handle,
    param_count: i32,
    param: [u32; 16],
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_fwspec_info {
    flags: u64,
    affinity: *const cpumask,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_ops {
    r#match: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, u32) -> i32>,
    select: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, u32) -> i32>,
    map: Option<unsafe extern "C" fn(*mut irq_domain, u32, u64) -> i32>,
    unmap: Option<unsafe extern "C" fn(*mut irq_domain, u32)>,
    xlate: Option<unsafe extern "C" fn(*mut irq_domain, *mut device_node, *const u32, u32, *mut u64, *mut u32) -> i32>,
    alloc: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32, *mut ()) -> i32>,
    free: Option<unsafe extern "C" fn(*mut irq_domain, u32, u32)>,
    activate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data, bool) -> i32>,
    deactivate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_data)>,
    translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
    get_fwspec_info: Option<unsafe extern "C" fn(*mut irq_fwspec, *mut irq_fwspec_info) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain {
    link: list_head,
    name: *const u8,
    ops: *const irq_domain_ops,
    host_data: *mut (),
    flags: u32,
    mapcount: u32,
    mutex: mutex,
    root: *mut irq_domain,
    fwnode: *mut fwnode_handle,
    bus_token: u32,
    gc: *mut irq_domain_chip_generic,
    dev: *mut device,
    pm_dev: *mut device,
    parent: *mut irq_domain,
    msi_parent_ops: *const msi_parent_ops,
    exit: Option<unsafe extern "C" fn(*mut irq_domain)>,
    hwirq_max: u64,
    revmap_size: u32,
    revmap_tree: xarray,
    revmap: [*mut irq_data; 0],
}


fn irq_domain_alloc_named_id_fwnode(name: *const u8, id: i32) -> *mut fwnode_handle {
unsafe { __irq_domain_alloc_fwnode(2u32, id, name, core::ptr::null_mut::<u64>(), core::ptr::null_mut::<fwnode_handle>()) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct irq_domain_info {
    fwnode: *mut fwnode_handle,
    domain_flags: u32,
    size: u32,
    hwirq_max: u64,
    direct_max: i32,
    hwirq_base: u32,
    virq_base: u32,
    bus_token: u32,
    name_suffix: *const u8,
    ops: *const irq_domain_ops,
    host_data: *mut (),
    dev: *mut device,
    parent: *mut irq_domain,
    dgc_info: *mut irq_domain_chip_generic_info,
    init: Option<unsafe extern "C" fn(*mut irq_domain) -> i32>,
    exit: Option<unsafe extern "C" fn(*mut irq_domain)>,
}


fn irq_resolve_mapping(domain: *mut irq_domain, hwirq: u64) -> *mut irq_desc {
unsafe { __irq_resolve_mapping(domain, hwirq, core::ptr::null_mut::<u32>()) }
}

fn irq_find_mapping(domain: *mut irq_domain, hwirq: u64) -> u32 {
let mut irq: u32 = 0;
if !unsafe { __irq_resolve_mapping(domain, hwirq, core::ptr::addr_of_mut!(irq)) }.is_null() {
return irq;
}
0
}

fn irq_domain_create_hierarchy(parent: *mut irq_domain, flags: u32, size: u32, fwnode: *mut fwnode_handle, ops: *const irq_domain_ops, host_data: *mut ()) -> *mut irq_domain {
let mut info: irq_domain_info = unsafe { core::mem::zeroed() };
let __v0: u32 = size;
info = {
let mut __t0: irq_domain_info = unsafe { core::mem::zeroed() };
__t0.fwnode = fwnode;
__t0.domain_flags = flags;
__t0.size = size;
__t0.hwirq_max = (if __v0 != 0 { __v0 } else { 4294967295u32 }) as u64;
__t0.ops = ops;
__t0.host_data = host_data;
__t0.parent = parent;
__t0
};
let d: *mut irq_domain = unsafe { irq_domain_instantiate(core::ptr::addr_of!(info)) };
if IS_ERR(d as *const ()) { core::ptr::null_mut::<irq_domain>() } else { d }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ioapic_domain_cfg {
    r#type: u32,
    ops: *const irq_domain_ops,
    dev: *mut device_node,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct x86_msi_data {
    __slate_anon_0: core::mem::MaybeUninit<[u8; 4]>,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct x86_msi_addr_lo {
    __slate_anon_0: __SlateRecord_msi_h_49,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct x86_msi_addr_hi {
    __slate_bits_0: __slate_bits::__SlateBits846U0,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_msg {
    __slate_anon_0: __SlateRecord_msi_h_62,
    __slate_anon_1: __SlateRecord_msi_h_66,
    __slate_anon_2: __SlateRecord_msi_h_70,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct pci_msi_desc {
    __slate_anon_0: __SlateRecord_msi_h_109,
    msi_attrib: __SlateRecord_msi_h_113,
    __slate_anon_2: __SlateRecord_msi_h_122,
}


#[repr(C)]
#[derive(Clone, Copy)]
union msi_domain_cookie {
    value: u64,
    ptr: *mut (),
    iobase: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc_data {
    dcookie: msi_domain_cookie,
    icookie: msi_instance_cookie,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_desc {
    irq: u32,
    nvec_used: u32,
    dev: *mut device,
    msg: msi_msg,
    affinity: *mut irq_affinity_desc,
    sysfs_attrs: *mut device_attribute,
    write_msi_msg: Option<unsafe extern "C" fn(*mut msi_desc, *mut ())>,
    write_msi_msg_data: *mut (),
    msi_index: u16,
    __slate_anon_9: __SlateRecord_msi_h_202,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_domain_ops {
    get_hwirq: Option<unsafe extern "C" fn(*mut msi_domain_info, *mut irq_alloc_info) -> u64>,
    msi_init: Option<unsafe extern "C" fn(*mut irq_domain, *mut msi_domain_info, u32, u64, *mut irq_alloc_info) -> i32>,
    msi_free: Option<unsafe extern "C" fn(*mut irq_domain, *mut msi_domain_info, u32)>,
    msi_prepare: Option<unsafe extern "C" fn(*mut irq_domain, *mut device, i32, *mut irq_alloc_info) -> i32>,
    msi_teardown: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_alloc_info)>,
    prepare_desc: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_alloc_info, *mut msi_desc)>,
    set_desc: Option<unsafe extern "C" fn(*mut irq_alloc_info, *mut msi_desc)>,
    domain_alloc_irqs: Option<unsafe extern "C" fn(*mut irq_domain, *mut device, i32) -> i32>,
    domain_free_irqs: Option<unsafe extern "C" fn(*mut irq_domain, *mut device)>,
    msi_translate: Option<unsafe extern "C" fn(*mut irq_domain, *mut irq_fwspec, *mut u64, *mut u32) -> i32>,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_domain_info {
    flags: u32,
    bus_token: u32,
    hwsize: u32,
    ops: *mut msi_domain_ops,
    dev: *mut device,
    chip: *mut irq_chip,
    chip_data: *mut (),
    handler: Option<unsafe extern "C" fn(*mut irq_desc)>,
    handler_data: *mut (),
    handler_name: *const u8,
    alloc_data: *mut irq_alloc_info,
    data: *mut (),
}


#[repr(C)]
#[derive(Clone, Copy)]
struct msi_parent_ops {
    supported_flags: u32,
    required_flags: u32,
    chip_flags: u32,
    bus_select_token: u32,
    bus_select_mask: u32,
    prefix: *const u8,
    init_dev_msi_info: Option<unsafe extern "C" fn(*mut device, *mut irq_domain, *mut irq_domain, *mut msi_domain_info) -> bool>,
}


#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.data")]
static mut no_timer_check: i32 = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct legacy_pic {
    nr_legacy_irqs: i32,
    chip: *mut irq_chip,
    mask: Option<unsafe extern "C" fn(u32)>,
    unmask: Option<unsafe extern "C" fn(u32)>,
    mask_all: Option<unsafe extern "C" fn()>,
    restore_mask: Option<unsafe extern "C" fn()>,
    init: Option<unsafe extern "C" fn(i32)>,
    probe: Option<unsafe extern "C" fn() -> i32>,
    irq_pending: Option<unsafe extern "C" fn(u32) -> i32>,
    make_irq: Option<unsafe extern "C" fn(u32)>,
}


fn nr_legacy_irqs() -> i32 {
unsafe { (*legacy_pic).nr_legacy_irqs }
}

#[repr(C)]
#[derive(Clone, Copy)]
union IO_APIC_reg_00 {
    raw: u32,
    bits: __SlateRecord_io_apic_h_21,
}


#[repr(C)]
#[derive(Clone, Copy)]
union IO_APIC_reg_01 {
    raw: u32,
    bits: __SlateRecord_io_apic_h_32,
}


#[repr(C)]
#[derive(Clone, Copy)]
union IO_APIC_reg_02 {
    raw: u32,
    bits: __SlateRecord_io_apic_h_43,
}


#[repr(C)]
#[derive(Clone, Copy)]
union IO_APIC_reg_03 {
    raw: u32,
    bits: __SlateRecord_io_apic_h_52,
}


#[repr(C, packed)]
#[derive(Clone, Copy)]
struct IO_APIC_route_entry {
    __slate_anon_0: __SlateRecord_io_apic_h_59,
}


#[unsafe(no_mangle)]
static mut nr_ioapics: i32 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut mp_irq_entries: i32 = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut mp_irqs: __SlateAlign16<[mpc_intsrc; 1024]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
#[unsafe(link_section = ".data..ro_after_init")]
static mut ioapic_is_disabled: bool = unsafe { core::mem::zeroed() };

#[unsafe(no_mangle)]
static mut gsi_top: u32 = unsafe { core::mem::zeroed() };

fn io_apic_read(apic__0: u32, reg: u32) -> u32 {
unsafe { unsafe { x86_apic_ops.io_apic_read }.unwrap()(apic__0, reg) }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct static_call_mod {
    next: *mut static_call_mod,
    r#mod: *mut module,
    sites: *mut static_call_site,
}


// Unconditional debug prints for code which is guarded by apic_verbosity already
fn apic_from_smp_config() -> bool {
(unsafe { smp_found_config }) != 0 && !unsafe { apic_is_disabled }
}

fn apic_is_x2apic_enabled() -> bool {
let mut msr: u64 = 0;
if rdmsrq_safe(27u32, unsafe { &mut (*core::ptr::addr_of_mut!(msr)) }) != 0 {
return false;
}
msr & BIT!(10i32) != 0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct apic {
    eoi: Option<unsafe extern "C" fn()>,
    native_eoi: Option<unsafe extern "C" fn()>,
    write: Option<unsafe extern "C" fn(u32, u32)>,
    read: Option<unsafe extern "C" fn(u32) -> u32>,
    wait_icr_idle: Option<unsafe extern "C" fn()>,
    safe_wait_icr_idle: Option<unsafe extern "C" fn() -> u32>,
    send_IPI: Option<unsafe extern "C" fn(i32, i32)>,
    send_IPI_mask: Option<unsafe extern "C" fn(*const cpumask, i32)>,
    send_IPI_mask_allbutself: Option<unsafe extern "C" fn(*const cpumask, i32)>,
    send_IPI_allbutself: Option<unsafe extern "C" fn(i32)>,
    send_IPI_all: Option<unsafe extern "C" fn(i32)>,
    send_IPI_self: Option<unsafe extern "C" fn(i32)>,
    __slate_bits_0: __slate_bits::__SlateBits875U0,
    calc_dest_apicid: Option<unsafe extern "C" fn(u32) -> u32>,
    icr_read: Option<unsafe extern "C" fn() -> u64>,
    icr_write: Option<unsafe extern "C" fn(u32, u32)>,
    max_apic_id: u32,
    probe: Option<unsafe extern "C" fn() -> i32>,
    setup: Option<unsafe extern "C" fn()>,
    teardown: Option<unsafe extern "C" fn()>,
    acpi_madt_oem_check: Option<unsafe extern "C" fn(*mut u8, *mut u8) -> i32>,
    init_apic_ldr: Option<unsafe extern "C" fn()>,
    cpu_present_to_apicid: Option<unsafe extern "C" fn(i32) -> u32>,
    get_apic_id: Option<unsafe extern "C" fn(u32) -> u32>,
    wakeup_secondary_cpu: Option<unsafe extern "C" fn(u32, u64, u32) -> i32>,
    wakeup_secondary_cpu_64: Option<unsafe extern "C" fn(u32, u64, u32) -> i32>,
    update_vector: Option<unsafe extern "C" fn(u32, u32, bool)>,
    name: *mut u8,
}


#[inline(always)]
fn apic_read(reg: u32) -> u32 {
let mut __v0: Option<unsafe extern "C" fn(u32) -> u32>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(u32) -> u32>>(__SCT__apic_call_read as *const ()) };
unsafe { __v0.unwrap()(reg) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__apic_call_read_605: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__apic_call_read) }) as *mut ();

#[inline(always)]
fn apic_write(reg: u32, val: u32) {
let mut __v0: Option<unsafe extern "C" fn(u32, u32)>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(u32, u32)>>(__SCT__apic_call_write as *const ()) };
unsafe { __v0.unwrap()(reg, val) };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__apic_call_write_606: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__apic_call_write) }) as *mut ();

#[inline(always)]
fn apic_eoi() {
let mut __v0: Option<unsafe extern "C" fn()>;
__v0 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn()>>(__SCT__apic_call_eoi as *const ()) };
unsafe { __v0.unwrap()() };
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__apic_call_eoi_607: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__apic_call_eoi) }) as *mut ();

fn read_apic_id() -> u32 {
let reg: u32 = apic_read(32u32);
unsafe { unsafe { (*apic).get_apic_id }.unwrap()(reg) }
}

// time_after()
static mut ioapic_lock: raw_spinlock = raw_spinlock { raw_lock: qspinlock { __slate_anon_0: {
let mut __t0: __SlateRecord_qspinlock_types_h_44 = unsafe { core::mem::zeroed() };
__t0.val = __SlateRecord_types_h_190 { counter: 0i32 };
__t0
} } };

static mut ioapic_mutex: mutex = {
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

static mut ioapic_dynirq_base: u32 = unsafe { core::mem::zeroed() };

static mut ioapic_initialized: i32 = unsafe { core::mem::zeroed() };

#[repr(C)]
#[derive(Clone, Copy)]
struct irq_pin_list {
    list: list_head,
    apic: i32,
    pin: i32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mp_chip_data {
    irq_2_pin: list_head,
    entry: IO_APIC_route_entry,
    is_level: bool,
    active_low: bool,
    isa_irq: bool,
    count: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct mp_ioapic_gsi {
    gsi_base: u32,
    gsi_end: u32,
}


#[repr(C)]
#[derive(Clone, Copy)]
struct ioapic {
    /// # of IRQ routing registers
    nr_registers: i32,
    /// Saved state during suspend/resume, or while enabling intr-remap.
    saved_registers: *mut IO_APIC_route_entry,
    /// I/O APIC config
    mp_config: mpc_ioapic,
    /// IO APIC gsi routing info
    gsi_config: mp_ioapic_gsi,
    irqdomain_cfg: ioapic_domain_cfg,
    irqdomain: *mut irq_domain,
    iomem_res: *mut resource,
}


static mut ioapics: __SlateAlign16<[ioapic; 128]> = __SlateAlign16(unsafe { core::mem::zeroed() });

#[unsafe(no_mangle)]
extern "C" fn mpc_ioapic_id(ioapic_idx: i32) -> i32 {
(unsafe { ioapics.0[ioapic_idx as usize].mp_config.apicid }) as i32
}

#[unsafe(no_mangle)]
extern "C" fn mpc_ioapic_addr(ioapic_idx: i32) -> u32 {
unsafe { ioapics.0[ioapic_idx as usize].mp_config.apicaddr }
}

fn mp_ioapic_gsi_routing(ioapic_idx: i32) -> *mut mp_ioapic_gsi {
unsafe { core::ptr::addr_of_mut!(ioapics.0[ioapic_idx as usize].gsi_config) }
}

fn mp_ioapic_pin_count(ioapic: i32) -> i32 {
let gsi_cfg: *mut mp_ioapic_gsi = mp_ioapic_gsi_routing(ioapic);
unsafe { (*gsi_cfg).gsi_end }.wrapping_sub(unsafe { (*gsi_cfg).gsi_base }).wrapping_add(1) as i32
}

fn mp_pin_to_gsi(ioapic: i32, pin: i32) -> u32 {
unsafe { (*mp_ioapic_gsi_routing(ioapic)).gsi_base }.wrapping_add(pin as u32)
}

fn mp_is_legacy_irq(irq: i32) -> bool {
irq >= 0 && irq < nr_legacy_irqs()
}

fn mp_ioapic_irqdomain(ioapic: i32) -> *mut irq_domain {
unsafe { ioapics.0[ioapic as usize].irqdomain }
}

// The one past the highest gsi number used
// MP IRQ source entries
/// # of MP IRQ source entries
/// disable_ioapic_support() - disables ioapic support at runtime
#[unsafe(no_mangle)]
extern "C" fn disable_ioapic_support() {
unsafe {
noioapicquirk = 1;
}
unsafe {
noioapicreroute = 1i32.wrapping_neg();
}
unsafe {
ioapic_is_disabled = true;
}
}

#[unsafe(link_section = ".init.text")]
extern "C" fn parse_noapic(str: *mut u8) -> i32 {
// disable IO-APIC
disable_ioapic_support();
0
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_parse_noapic: [u8; 7] = [110, 111, 97, 112, 105, 99, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_parse_noapic: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_parse_noapic) as *const u8 };
__t0.setup_func = Some(parse_noapic);
__t0.early = 1i32;
__t0
};

/// Will be called in mpparse/ACPI codes for saving IRQ info
#[unsafe(no_mangle)]
extern "C" fn mp_save_irq(m: *mut mpc_intsrc) {
let mut i: i32 = 0;
loop {
if 1 <= unsafe { apic_verbosity } {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016Int: type %d, pol %d, trig %d, bus %02x, IRQ %02x, APIC ID %x, APIC INT %02x\n".as_ptr() as *const u8, (*m).irqtype as i32, ((*m).irqflag as i32) & 3, ((*m).irqflag as i32) >> 2i32 & 3, (*m).srcbus as i32, (*m).srcbusirq as i32, (*m).dstapic as i32, (*m).dstirq as i32) };
}
if true {
break;
}
}
while i < unsafe { mp_irq_entries } {
if (unsafe { memcmp((core::ptr::addr_of_mut!(mp_irqs.0) as *mut mpc_intsrc).wrapping_offset(i as isize) as *const (), m as *const (), core::mem::size_of::<mpc_intsrc>() as u64) }) == 0 {
return;
}
i = i.wrapping_add(1);
}
unsafe { memcpy((core::ptr::addr_of_mut!(mp_irqs.0) as *mut mpc_intsrc).wrapping_offset(mp_irq_entries as isize) as *mut (), m as *const (), core::mem::size_of::<mpc_intsrc>() as u64) };
let __v6: i32 = unsafe { mp_irq_entries };
let __v7: i32 = __v6.wrapping_add(1);
unsafe {
mp_irq_entries = __v7;
}
if __v7 == 1024i32 {
unsafe { panic(c"Max # of irq sources exceeded!!\n".as_ptr() as *const u8) };
unsafe { core::hint::unreachable_unchecked() };
}
}

fn alloc_ioapic_saved_registers(idx: i32) {
let mut __slate_compound_2: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_4: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut size: u64 = 0;
if !unsafe { ioapics.0[idx as usize].saved_registers }.is_null() {
return;
}
size = (core::mem::size_of::<IO_APIC_route_entry>() as u64).wrapping_mul((unsafe { ioapics.0[idx as usize].nr_registers }) as u64);
let mut __v0: *mut ();
let mut __v1: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kzalloc_noprof(size, 3264u32, unsafe { *{
__slate_compound_2 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_2)
} });
alloc_tag_restore!();
} else {
_res = _kzalloc_noprof(size, 3264u32, unsafe { *{
__slate_compound_4 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_4)
} });
}
__v1 = _res;
__v0 = __v1;
unsafe {
ioapics.0[idx as usize].saved_registers = __v0 as *mut IO_APIC_route_entry;
}
if unsafe { ioapics.0[idx as usize].saved_registers }.is_null() {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x013IOAPIC %d: suspend/resume impossible!\n".as_ptr() as *const u8, idx) };
}
}

fn free_ioapic_saved_registers(idx: i32) {
unsafe { kfree(ioapics.0[idx as usize].saved_registers as *const ()) };
unsafe {
ioapics.0[idx as usize].saved_registers = core::ptr::null_mut::<IO_APIC_route_entry>();
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn arch_early_ioapic_init() -> i32 {
let mut i: i32 = 0;
if nr_legacy_irqs() == 0 {
unsafe {
io_apic_irqs = !0;
}
}
while i < unsafe { nr_ioapics } {
alloc_ioapic_saved_registers(i);
i = i.wrapping_add(1);
}
0
}

#[repr(C)]
#[derive(Clone, Copy)]
struct io_apic {
    index: u32,
    unused: [u32; 3],
    data: u32,
    unused2: [u32; 11],
    eoi: u32,
}


fn io_apic_base(idx: i32) -> *mut io_apic {
((unsafe { ((__fix_to_virt!(516i32.wrapping_add(idx)) as *mut ()) as *const u8).wrapping_offset(((mpc_ioapic_addr(idx) as u64) & !(!(_AC!() << 12i32).wrapping_sub(1))) as isize) }) as *mut ()) as *mut io_apic
}

fn io_apic_eoi(apic__0: u32, vector: u32) {
let io_apic: *mut io_apic = io_apic_base(apic__0 as i32);
writel(vector, (unsafe { core::ptr::addr_of_mut!((*io_apic).eoi) }) as *mut ());
}

#[unsafe(no_mangle)]
extern "C" fn native_io_apic_read(apic__0: u32, reg: u32) -> u32 {
let io_apic: *mut io_apic = io_apic_base(apic__0 as i32);
writel(reg, (unsafe { core::ptr::addr_of_mut!((*io_apic).index) }) as *mut ());
readl((unsafe { core::ptr::addr_of_mut!((*io_apic).data) }) as *const ())
}

fn io_apic_write(apic__0: u32, reg: u32, value: u32) {
let io_apic: *mut io_apic = io_apic_base(apic__0 as i32);
writel(reg, (unsafe { core::ptr::addr_of_mut!((*io_apic).index) }) as *mut ());
writel(value, (unsafe { core::ptr::addr_of_mut!((*io_apic).data) }) as *mut ());
}

fn __ioapic_read_entry(apic__0: i32, pin: i32) -> IO_APIC_route_entry {
let mut entry: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let __t0 = io_apic_read(apic__0 as u32, 16i32.wrapping_add(2i32.wrapping_mul(pin)) as u32) as u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_2.__slate_bits_0));
__t1.__set_w1(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_2.__slate_bits_0), __t1);
}
let __t2 = io_apic_read(apic__0 as u32, 17i32.wrapping_add(2i32.wrapping_mul(pin)) as u32) as u64;
unsafe {
let mut __t3 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_2.__slate_bits_0));
__t3.__set_w2(__t2);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_2.__slate_bits_0), __t3);
}
return entry;
}

fn ioapic_read_entry(apic__0: i32, pin: i32) -> IO_APIC_route_entry {
let mut __UNIQUE_ID_guard_620: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_620) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_621: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_621) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
return __ioapic_read_entry(apic__0, pin);
}

/// When we write a new IO APIC routing entry, we need to write the high
/// word first! If the mask bit in the low word is clear, we will enable
/// the interrupt, and we need to make sure the entry is fully populated
/// before that happens.
fn __ioapic_write_entry(apic__0: i32, pin: i32, e: IO_APIC_route_entry) {
io_apic_write(apic__0 as u32, 17i32.wrapping_add(2i32.wrapping_mul(pin)) as u32, (unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(e.__slate_anon_0.__slate_anon_2.__slate_bits_0)).__get_w2() }) as u32);
io_apic_write(apic__0 as u32, 16i32.wrapping_add(2i32.wrapping_mul(pin)) as u32, (unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(e.__slate_anon_0.__slate_anon_2.__slate_bits_0)).__get_w1() }) as u32);
}

fn ioapic_write_entry(apic__0: i32, pin: i32, e: IO_APIC_route_entry) {
let mut __UNIQUE_ID_guard_622: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_622) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_623: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_623) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
__ioapic_write_entry(apic__0, pin, e);
}

/// When we mask an IO APIC routing entry, we need to write the low
/// word first, in order to set the mask bit before we change the
/// high bits!
fn ioapic_mask_entry(apic__0: i32, pin: i32) {
let e: IO_APIC_route_entry = IO_APIC_route_entry { __slate_anon_0: {
let mut __t0: __SlateRecord_io_apic_h_59 = unsafe { core::mem::zeroed() };
__t0.__slate_anon_0 = {
let mut __t1: __SlateRecord_io_apic_h_60 = unsafe { core::mem::zeroed() };
__t1.__slate_bits_0.__set_masked(1u64);
__t1
};
__t0
} };
let mut __UNIQUE_ID_guard_624: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_624) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_625: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_625) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
io_apic_write(apic__0 as u32, 16i32.wrapping_add(2i32.wrapping_mul(pin)) as u32, (unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(e.__slate_anon_0.__slate_anon_2.__slate_bits_0)).__get_w1() }) as u32);
io_apic_write(apic__0 as u32, 17i32.wrapping_add(2i32.wrapping_mul(pin)) as u32, (unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(e.__slate_anon_0.__slate_anon_2.__slate_bits_0)).__get_w2() }) as u32);
}

/// The common case is 1:1 IRQ<->pin mappings. Sometimes there are
/// shared ISA-space IRQs, so we have to support them. We are super
/// fast in the common case, and fast for shared ISA-space IRQs.
fn add_pin_to_irq_node(data: *mut mp_chip_data, node: i32, apic__0: i32, pin: i32) -> bool {
let mut __slate_compound_5: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_7: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut entry: *mut irq_pin_list = unsafe { core::mem::zeroed() };
// Don't allow duplicates
entry = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) });
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }, list) == 0 {
if (unsafe { (*entry).apic }) == apic__0 && (unsafe { (*entry).pin }) == pin {
return true;
}
entry = list_next_entry__2!(entry, list);
}
let mut __v3: *mut ();
let mut __v4: *mut ();
let mut _res: *mut () = unsafe { core::mem::zeroed() };
if mem_alloc_profiling_enabled() {
core::ptr::null_mut::<alloc_tag>();
_res = _kmalloc_node_noprof(core::mem::size_of::<irq_pin_list>() as u64, 2336u32, node, unsafe { *{
__slate_compound_5 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_5)
} });
alloc_tag_restore!();
} else {
_res = _kmalloc_node_noprof(core::mem::size_of::<irq_pin_list>() as u64, 2336u32, node, unsafe { *{
__slate_compound_7 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_7)
} });
}
__v4 = _res;
__v3 = __v4;
entry = __v3 as *mut irq_pin_list;
if entry.is_null() {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x013Cannot allocate irq_pin_list (%d,%d,%d)\n".as_ptr() as *const u8, node, apic__0, pin) };
return false;
}
unsafe {
(*entry).apic = apic__0;
}
unsafe {
(*entry).pin = pin;
}
list_add_tail(unsafe { core::ptr::addr_of_mut!((*entry).list) }, unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) });
true
}

fn __remove_pin_from_irq(data: *mut mp_chip_data, apic__0: i32, pin: i32) {
let mut tmp: *mut irq_pin_list = unsafe { core::mem::zeroed() };
let mut entry: *mut irq_pin_list = unsafe { core::mem::zeroed() };
entry = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) });
tmp = list_next_entry__2!(entry, list);
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }, list) == 0 {
if (unsafe { (*entry).apic }) == apic__0 && (unsafe { (*entry).pin }) == pin {
list_del(unsafe { core::ptr::addr_of_mut!((*entry).list) });
unsafe { kfree(entry as *const ()) };
return;
}
entry = tmp;
tmp = list_next_entry__2!(tmp, list);
}
}

fn io_apic_modify_irq(data: *mut mp_chip_data, masked: bool, r#final: Option<unsafe extern "C" fn(*mut irq_pin_list)>) {
let mut entry: *mut irq_pin_list = unsafe { core::mem::zeroed() };
let __t0 = masked as u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*data).entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t1.__set_masked(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*data).entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t1);
}
entry = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) });
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }, list) == 0 {
io_apic_write((unsafe { (*entry).apic }) as u32, 16i32.wrapping_add(2i32.wrapping_mul(unsafe { (*entry).pin })) as u32, (unsafe { core::ptr::read_unaligned(core::ptr::addr_of!((*data).entry.__slate_anon_0.__slate_anon_2.__slate_bits_0)).__get_w1() }) as u32);
if r#final != None {
unsafe { r#final.unwrap()(entry) };
}
entry = list_next_entry__2!(entry, list);
}
}

/// Synchronize the IO-APIC and the CPU by doing a dummy read from the
/// IO-APIC
#[unsafe(link_section = ".text.slate_distinct.io_apic.io_apic_sync")]
extern "C" fn io_apic_sync(entry: *mut irq_pin_list) {
let mut io_apic: *mut io_apic = unsafe { core::mem::zeroed() };
io_apic = io_apic_base(unsafe { (*entry).apic });
readl((unsafe { core::ptr::addr_of_mut!((*io_apic).data) }) as *const ());
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.mask_ioapic_irq")]
extern "C" fn mask_ioapic_irq(irq_data: *mut irq_data) {
let data: *mut mp_chip_data = (unsafe { (*irq_data).chip_data }) as *mut mp_chip_data;
let mut __UNIQUE_ID_guard_626: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_626) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_627: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_627) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
io_apic_modify_irq(data, true, Some(io_apic_sync));
}

fn __unmask_ioapic(data: *mut mp_chip_data) {
io_apic_modify_irq(data, false, None);
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.unmask_ioapic_irq")]
extern "C" fn unmask_ioapic_irq(irq_data: *mut irq_data) {
let data: *mut mp_chip_data = (unsafe { (*irq_data).chip_data }) as *mut mp_chip_data;
let mut __UNIQUE_ID_guard_628: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_628) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_629: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_629) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
__unmask_ioapic(data);
}

/// IO-APIC versions below 0x20 don't support EOI register.
/// For the record, here is the information about various versions:
///     0Xh     82489DX
///     1Xh     I/OAPIC or I/O(x)APIC which are not PCI 2.2 Compliant
///     2Xh     I/O(x)APIC which is PCI 2.2 Compliant
///     30h-FFh Reserved
///
/// Some of the Intel ICH Specs (ICH2 to ICH5) documents the io-apic
/// version as 0x2. This is an error with documentation and these ICH chips
/// use io-apic's of version 0x20.
///
/// For IO-APIC's with EOI register, we use that to do an explicit EOI.
/// Otherwise, we simulate the EOI message manually by changing the trigger
/// mode to edge and then back to level, with RTE being masked during this.
fn __eoi_ioapic_pin(apic__0: i32, pin: i32, vector: i32) {
if (mpc_ioapic_ver!(apic__0) as i32) >= 32 {
io_apic_eoi(apic__0 as u32, vector as u32);
} else {
let mut entry: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut entry1: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry1) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let __v0: IO_APIC_route_entry = __ioapic_read_entry(apic__0, pin);
entry1 = __v0;
entry = __v0;
// Mask the entry and change the trigger mode to edge.
let __t0 = 1u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t1.__set_masked(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t1);
}
let __t2 = 0u64;
unsafe {
let mut __t3 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t3.__set_is_level(__t2);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t3);
}
__ioapic_write_entry(apic__0, pin, entry1);
// Restore the previous level triggered entry.
__ioapic_write_entry(apic__0, pin, entry);
}
}

fn eoi_ioapic_pin(vector: i32, data: *mut mp_chip_data) {
let mut entry: *mut irq_pin_list = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_630: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_630) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_631: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_631) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
entry = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) });
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }, list) == 0 {
__eoi_ioapic_pin(unsafe { (*entry).apic }, unsafe { (*entry).pin }, vector);
entry = list_next_entry__2!(entry, list);
}
}

fn clear_IO_APIC_pin(apic__0: u32, pin: u32) {
let mut entry: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
// Check delivery_mode to be sure we're not clearing an SMI pin
entry = ioapic_read_entry(apic__0 as i32, pin as i32);
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_delivery_mode() }) as i32) == 2 {
return;
}
// Make sure the entry is masked and re-read the contents to check
// if it is a level triggered pin and if the remote-IRR is set.
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_masked() }) as i32) == 0 {
let __t0 = 1u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t1.__set_masked(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t1);
}
ioapic_write_entry(apic__0 as i32, pin as i32, entry);
entry = ioapic_read_entry(apic__0 as i32, pin as i32);
}
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_irr() }) as i32) != 0 {
// Make sure the trigger mode is set to level. Explicit EOI
// doesn't clear the remote-IRR if the trigger mode is not
// set to level.
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_is_level() }) as i32) == 0 {
let __t2 = 1u64;
unsafe {
let mut __t3 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t3.__set_is_level(__t2);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t3);
}
ioapic_write_entry(apic__0 as i32, pin as i32, entry);
}
let mut __UNIQUE_ID_guard_632: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_632) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_633: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_633) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
__eoi_ioapic_pin(apic__0 as i32, pin as i32, (unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_vector() }) as i32);
}
// Clear the rest of the bits in the IO-APIC RTE except for the mask
// bit.
ioapic_mask_entry(apic__0 as i32, pin as i32);
entry = ioapic_read_entry(apic__0 as i32, pin as i32);
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_irr() }) as i32) != 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x013Unable to reset IRR for apic: %d, pin :%d\n".as_ptr() as *const u8, mpc_ioapic_id(apic__0 as i32), pin) };
}
}

#[unsafe(no_mangle)]
extern "C" fn clear_IO_APIC() {
let mut apic__0: i32 = 0;
let mut pin: i32 = 0;
while apic__0 < unsafe { nr_ioapics } {
pin = 0;
while pin < unsafe { ioapics.0[apic__0 as usize].nr_registers } {
clear_IO_APIC_pin(apic__0 as u32, pin as u32);
pin = pin.wrapping_add(1);
}
apic__0 = apic__0.wrapping_add(1);
}
}

/// Saves all the IO-APIC RTE's
#[unsafe(no_mangle)]
extern "C" fn save_ioapic_entries() -> i32 {
let mut apic__0: i32 = 0;
let mut pin: i32 = 0;
let mut err: i32 = 0;
while apic__0 < unsafe { nr_ioapics } {
if unsafe { ioapics.0[apic__0 as usize].saved_registers }.is_null() {
err = 12i32.wrapping_neg();
} else {
pin = 0;
while pin < unsafe { ioapics.0[apic__0 as usize].nr_registers } {
unsafe {
*unsafe { ioapics.0[apic__0 as usize].saved_registers }.wrapping_offset(pin as isize) = ioapic_read_entry(apic__0, pin);
}
pin = pin.wrapping_add(1);
}
}
apic__0 = apic__0.wrapping_add(1);
}
err
}

/// Mask all IO APIC entries.
#[unsafe(no_mangle)]
extern "C" fn mask_ioapic_entries() {
let mut apic__0: i32 = 0;
let mut pin: i32 = 0;
while apic__0 < unsafe { nr_ioapics } {
if !unsafe { ioapics.0[apic__0 as usize].saved_registers }.is_null() {
pin = 0;
while pin < unsafe { ioapics.0[apic__0 as usize].nr_registers } {
let mut entry: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
entry = unsafe { *unsafe { ioapics.0[apic__0 as usize].saved_registers }.wrapping_offset(pin as isize) };
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_masked() }) as i32) == 0 {
let __t0 = 1u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t1.__set_masked(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t1);
}
ioapic_write_entry(apic__0, pin, entry);
}
pin = pin.wrapping_add(1);
}
}
apic__0 = apic__0.wrapping_add(1);
}
}

/// Restore IO APIC entries which was saved in the ioapic structure.
#[unsafe(no_mangle)]
extern "C" fn restore_ioapic_entries() -> i32 {
let mut apic__0: i32 = 0;
let mut pin: i32 = 0;
while apic__0 < unsafe { nr_ioapics } {
if !unsafe { ioapics.0[apic__0 as usize].saved_registers }.is_null() {
pin = 0;
while pin < unsafe { ioapics.0[apic__0 as usize].nr_registers } {
ioapic_write_entry(apic__0, pin, unsafe { *unsafe { ioapics.0[apic__0 as usize].saved_registers }.wrapping_offset(pin as isize) });
pin = pin.wrapping_add(1);
}
}
apic__0 = apic__0.wrapping_add(1);
}
0
}

/// Find the IRQ entry number of a certain pin.
fn find_irq_entry(ioapic_idx: i32, pin: i32, r#type: i32) -> i32 {
let mut i: i32 = 0;
while i < unsafe { mp_irq_entries } {
let __v0: bool = ((unsafe { mp_irqs.0[i as usize].irqtype }) as i32) == r#type && (((unsafe { mp_irqs.0[i as usize].dstapic }) as i32) == mpc_ioapic_id(ioapic_idx) || ((unsafe { mp_irqs.0[i as usize].dstapic }) as i32) == 255);
if __v0 && ((unsafe { mp_irqs.0[i as usize].dstirq }) as i32) == pin {
return i;
}
i = i.wrapping_add(1);
}
1i32.wrapping_neg()
}

/// Find the pin to which IRQ[irq] (ISA) is connected
#[unsafe(link_section = ".init.text")]
fn find_isa_irq_pin(irq: i32, r#type: i32) -> i32 {
let mut i: i32 = 0;
while i < unsafe { mp_irq_entries } {
let lbus: i32 = (unsafe { mp_irqs.0[i as usize].srcbus }) as i32;
if test_bit!(lbus, unsafe { core::ptr::addr_of_mut!(mp_bus_not_pci.0) as *mut u64 }) != 0 && ((unsafe { mp_irqs.0[i as usize].irqtype }) as i32) == r#type && ((unsafe { mp_irqs.0[i as usize].srcbusirq }) as i32) == irq {
return (unsafe { mp_irqs.0[i as usize].dstirq }) as i32;
}
i = i.wrapping_add(1);
}
1i32.wrapping_neg()
}

#[unsafe(link_section = ".init.text")]
fn find_isa_irq_apic(irq: i32, r#type: i32) -> i32 {
let mut i: i32 = 0;
while i < unsafe { mp_irq_entries } {
let lbus: i32 = (unsafe { mp_irqs.0[i as usize].srcbus }) as i32;
if test_bit!(lbus, unsafe { core::ptr::addr_of_mut!(mp_bus_not_pci.0) as *mut u64 }) != 0 && ((unsafe { mp_irqs.0[i as usize].irqtype }) as i32) == r#type && ((unsafe { mp_irqs.0[i as usize].srcbusirq }) as i32) == irq {
break;
}
i = i.wrapping_add(1);
}
if i < unsafe { mp_irq_entries } {
let mut ioapic_idx: i32 = 0;
while ioapic_idx < unsafe { nr_ioapics } {
if mpc_ioapic_id(ioapic_idx) == ((unsafe { mp_irqs.0[i as usize].dstapic }) as i32) {
return ioapic_idx;
}
ioapic_idx = ioapic_idx.wrapping_add(1);
}
}
1i32.wrapping_neg()
}

fn irq_active_low(idx: i32) -> bool {
let bus: i32 = (unsafe { mp_irqs.0[idx as usize].srcbus }) as i32;
// Determine IRQ line polarity (high active or low active):
// Conforms to spec, ie. bus-type dependent polarity.  PCI
// defaults to low active. [E]ISA defaults to high active.
// Pointless default required due to do gcc stupidity
match ((unsafe { mp_irqs.0[idx as usize].irqflag }) as i32) & 3 {
0 => {
return test_bit!(bus, unsafe { core::ptr::addr_of_mut!(mp_bus_not_pci.0) as *mut u64 }) == 0;
// Conforms to spec, ie. bus-type dependent polarity.  PCI
// defaults to low active. [E]ISA defaults to high active.
}
1 => {
return false;
}
2 => {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014IOAPIC: Invalid polarity: 2, defaulting to low\n".as_ptr() as *const u8) };
return true; // Pointless default required due to do gcc stupidity
}
_ => {
return true; // Pointless default required due to do gcc stupidity
}
}
unsafe { core::mem::zeroed() }
}

fn eisa_irq_is_level(idx: i32, bus: i32, level: bool) -> i32 {
level as i32
}

fn irq_is_level(idx: i32) -> bool {
let bus: i32 = (unsafe { mp_irqs.0[idx as usize].srcbus }) as i32;
let mut level: bool = false;
// Determine IRQ trigger mode (edge or level sensitive):
// Conforms to spec, ie. bus-type dependent trigger
// mode. PCI defaults to level, ISA to edge.
// Take EISA into account
// Pointless default required due to do gcc stupidity
match ((unsafe { mp_irqs.0[idx as usize].irqflag }) as i32) & 12 {
0 => {
level = test_bit!(bus, unsafe { core::ptr::addr_of_mut!(mp_bus_not_pci.0) as *mut u64 }) == 0;
// Conforms to spec, ie. bus-type dependent trigger
// mode. PCI defaults to level, ISA to edge.
// Take EISA into account
return eisa_irq_is_level(idx, bus, level) != 0;
}
4 => {
return false;
}
8 => {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014IOAPIC: Invalid trigger mode 2 defaulting to level\n".as_ptr() as *const u8) };
return true; // Pointless default required due to do gcc stupidity
}
_ => {
return true; // Pointless default required due to do gcc stupidity
}
}
unsafe { core::mem::zeroed() }
}

fn __acpi_get_override_irq(gsi: u32, trigger: *mut bool, polarity: *mut bool) -> i32 {
let mut ioapic: i32 = 0;
let mut pin: i32 = 0;
let mut idx: i32 = 0;
if unsafe { ioapic_is_disabled } {
return 1i32.wrapping_neg();
}
ioapic = mp_find_ioapic(gsi);
if ioapic < 0 {
return 1i32.wrapping_neg();
}
pin = mp_find_ioapic_pin(ioapic, gsi);
if pin < 0 {
return 1i32.wrapping_neg();
}
idx = find_irq_entry(ioapic, pin, 0i32);
if idx < 0 {
return 1i32.wrapping_neg();
}
unsafe {
*trigger = irq_is_level(idx);
}
unsafe {
*polarity = irq_active_low(idx);
}
0
}

#[unsafe(no_mangle)]
extern "C" fn acpi_get_override_irq(gsi: u32, is_level: *mut i32, active_low: *mut i32) -> i32 {
unsafe {
*active_low = 0;
}
unsafe {
*is_level = 0;
}
__acpi_get_override_irq(gsi, is_level as *mut bool, active_low as *mut bool)
}

#[unsafe(no_mangle)]
extern "C" fn ioapic_set_alloc_attr(info: *mut irq_alloc_info, node: i32, trigger: i32, polarity: i32) {
unsafe { init_irq_alloc_info(info, core::ptr::null::<cpumask>()) };
unsafe {
(*info).r#type = 1;
}
unsafe {
(*info).__slate_anon_7.ioapic.node = node;
}
unsafe {
(*info).__slate_anon_7.ioapic.__slate_bits_0.__set_is_level(trigger as u32);
}
unsafe {
(*info).__slate_anon_7.ioapic.__slate_bits_0.__set_active_low(polarity as u32);
}
unsafe {
(*info).__slate_anon_7.ioapic.__slate_bits_0.__set_valid(1u32);
}
}

fn ioapic_copy_alloc_attr(dst: *mut irq_alloc_info, src: *mut irq_alloc_info, gsi: u32, ioapic_idx: i32, pin: i32) {
let mut level: bool = false;
let mut pol_low: bool = false;
unsafe { copy_irq_alloc_info(dst, src) };
unsafe {
(*dst).r#type = 1;
}
unsafe {
(*dst).devid = mpc_ioapic_id(ioapic_idx) as u32;
}
unsafe {
(*dst).__slate_anon_7.ioapic.pin = pin;
}
unsafe {
(*dst).__slate_anon_7.ioapic.__slate_bits_0.__set_valid(1u32);
}
if !src.is_null() && ((unsafe { (*src).__slate_anon_7.ioapic.__slate_bits_0.__get_valid() }) as i32) != 0 {
unsafe {
(*dst).__slate_anon_7.ioapic.node = (*src).__slate_anon_7.ioapic.node;
}
unsafe {
let __t0 = ((*src).__slate_anon_7.ioapic.__slate_bits_0.__get_is_level() as i32) as u32;
(*dst).__slate_anon_7.ioapic.__slate_bits_0.__set_is_level(__t0);
}
unsafe {
let __t1 = ((*src).__slate_anon_7.ioapic.__slate_bits_0.__get_active_low() as i32) as u32;
(*dst).__slate_anon_7.ioapic.__slate_bits_0.__set_active_low(__t1);
}
} else {
unsafe {
(*dst).__slate_anon_7.ioapic.node = 1i32.wrapping_neg();
}
if __acpi_get_override_irq(gsi, core::ptr::addr_of_mut!(level), core::ptr::addr_of_mut!(pol_low)) >= 0 {
unsafe {
(*dst).__slate_anon_7.ioapic.__slate_bits_0.__set_is_level(level as u32);
}
unsafe {
(*dst).__slate_anon_7.ioapic.__slate_bits_0.__set_active_low(pol_low as u32);
}
} else {
// PCI interrupts are always active low level
// triggered.
unsafe {
(*dst).__slate_anon_7.ioapic.__slate_bits_0.__set_is_level(1u32);
}
unsafe {
(*dst).__slate_anon_7.ioapic.__slate_bits_0.__set_active_low(1u32);
}
}
}
}

fn ioapic_alloc_attr_node(info: *mut irq_alloc_info) -> i32 {
if !info.is_null() && ((unsafe { (*info).__slate_anon_7.ioapic.__slate_bits_0.__get_valid() }) as i32) != 0 { unsafe { (*info).__slate_anon_7.ioapic.node } } else { 1i32.wrapping_neg() }
}

fn mp_register_handler(irq: u32, level: bool) {
let fasteoi: bool = if level {
irq_set_status_flags(irq, 256u64);
true
} else {
irq_clear_status_flags(irq, 256u64);
false
};
let hdl: Option<unsafe extern "C" fn(*mut irq_desc)> = {
let __t0: Option<unsafe extern "C" fn(*mut irq_desc)> = if fasteoi { unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_desc)>>(handle_fasteoi_irq as *const ()) } } else { unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_desc)>>(handle_edge_irq as *const ()) } };
__t0
};
unsafe { __irq_set_handler(irq, hdl, 0i32, (if fasteoi { c"fasteoi".as_ptr() as *mut u8 } else { c"edge".as_ptr() as *mut u8 }) as *const u8) };
}

fn mp_check_pin_attr(irq: i32, info: *mut irq_alloc_info) -> bool {
let data: *mut mp_chip_data = irq_get_chip_data(irq as u32) as *mut mp_chip_data;
// setup_IO_APIC_irqs() programs all legacy IRQs with default trigger
// and polarity attributes. So allow the first user to reprogram the
// pin with real trigger and polarity attributes.
if irq < nr_legacy_irqs() && (unsafe { (*data).count }) == 1 {
if ((unsafe { (*info).__slate_anon_7.ioapic.__slate_bits_0.__get_is_level() }) as i32) != ((unsafe { (*data).is_level }) as i32) {
mp_register_handler(irq as u32, ((unsafe { (*info).__slate_anon_7.ioapic.__slate_bits_0.__get_is_level() }) as i32) != 0);
}
let __v0: bool = ((unsafe { (*info).__slate_anon_7.ioapic.__slate_bits_0.__get_is_level() }) as i32) != 0;
unsafe {
(*data).is_level = __v0;
}
let __t0 = __v0 as u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*data).entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t1.__set_is_level(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*data).entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t1);
}
let __v1: bool = ((unsafe { (*info).__slate_anon_7.ioapic.__slate_bits_0.__get_active_low() }) as i32) != 0;
unsafe {
(*data).active_low = __v1;
}
let __t2 = __v1 as u64;
unsafe {
let mut __t3 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*data).entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t3.__set_active_low(__t2);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*data).entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t3);
}
}
((unsafe { (*data).is_level }) as i32) == ((unsafe { (*info).__slate_anon_7.ioapic.__slate_bits_0.__get_is_level() }) as i32) && ((unsafe { (*data).active_low }) as i32) == ((unsafe { (*info).__slate_anon_7.ioapic.__slate_bits_0.__get_active_low() }) as i32)
}

fn alloc_irq_from_domain(domain: *mut irq_domain, ioapic: i32, gsi: u32, info: *mut irq_alloc_info) -> i32 {
let r#type: i32 = ((unsafe { ioapics.0[ioapic as usize].irqdomain_cfg.r#type }) as u32) as i32;
let mut legacy: bool = false;
let mut irq: i32 = 1i32.wrapping_neg();
match r#type {
1 => {
let __v0: bool = (unsafe { ioapic_initialized }) == 0 || gsi >= (nr_legacy_irqs() as u32);
if __v0 {
irq = gsi as i32;
}
// Dynamically allocate IRQ number for non-ISA IRQs in the first
// 16 GSIs on some weird platforms.
legacy = mp_is_legacy_irq(irq);
}
2 => {
irq = gsi as i32;
}
3 => {
}
_ => {
let mut __v2: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
BUGFLAG_TAINT!(9i32) | 1i32 | 16i32;
unsafe { __WARN_validate_printf(c"ioapic: unknown irqdomain type %d\n".as_ptr() as *const u8, r#type) };
let mut __v3: Option<unsafe extern "C" fn(*mut bug_entry, ...)>;
__v3 = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut bug_entry, ...)>>(__SCT__WARN_trap as *const ()) };
let mut __v4: *mut bug_entry;
let mut bug: *mut bug_entry = unsafe { core::mem::zeroed() };
unsafe {
let __t0: u64;
core::arch::asm!("lea (2f)(%rip), {0:r}\n1:\n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {1} - .\t# bug_entry::format\n\t.long {2} - .\t# bug_entry::file\n\t.word {3}\t# bug_entry::line\n\t.word {4}\t# bug_entry::flags\n\t.org 2b + {5}\n.popsection\n", lateout(reg) __t0, sym __slate_asm_string_alloc_irq_from_domain_5, sym __slate_asm_string_alloc_irq_from_domain_6, const 905 as i32, const 2321 as i32, const 16 as u64, options(nostack, att_syntax));
bug = unsafe { core::mem::transmute::<u64, *mut bug_entry>(__t0 as u64) };
}
__v4 = bug;
unsafe { __v3.unwrap()(__v4, r#type) };
unsafe {
core::arch::asm!("", options(att_syntax));
}
if true {
break;
}
}
}
__v2 = (__ret_warn_on != 0) as i64;
return 1i32.wrapping_neg();
}
}
unsafe { __irq_domain_alloc_irqs(domain, irq, 1u32, ioapic_alloc_attr_node(info), info as *mut (), legacy, core::ptr::null::<irq_affinity_desc>()) }
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable___SCK__WARN_trap_634: *mut () = (unsafe { core::ptr::addr_of_mut!(__SCK__WARN_trap) }) as *mut ();

/// Need special handling for ISA IRQs because there may be multiple IOAPIC pins
/// sharing the same ISA IRQ number and irqdomain only supports 1:1 mapping
/// between IOAPIC pin and IRQ number. A typical IOAPIC has 24 pins, pin 0-15 are
/// used for legacy IRQs and pin 16-23 are used for PCI IRQs (PIRQ A-H).
/// When ACPI is disabled, only legacy IRQ numbers (IRQ0-15) are available, and
/// some BIOSes may use MP Interrupt Source records to override IRQ numbers for
/// PIRQs instead of reprogramming the interrupt routing logic. Thus there may be
/// multiple pins sharing the same legacy IRQ number when ACPI is disabled.
fn alloc_isa_irq_from_domain(domain: *mut irq_domain, mut irq: i32, ioapic: i32, pin: i32, info: *mut irq_alloc_info) -> i32 {
let mut irq_data: *mut irq_data = unsafe { irq_get_irq_data(irq as u32) };
let node: i32 = ioapic_alloc_attr_node(info);
// Legacy ISA IRQ has already been allocated, just add pin to
// the pin list associated with this IRQ and program the IOAPIC
// entry.
if !irq_data.is_null() && !unsafe { (*irq_data).parent_data }.is_null() {
if !mp_check_pin_attr(irq, info) {
return 16i32.wrapping_neg();
}
if !add_pin_to_irq_node((unsafe { (*irq_data).chip_data }) as *mut mp_chip_data, node, ioapic, unsafe { (*info).__slate_anon_7.ioapic.pin }) {
return 12i32.wrapping_neg();
}
} else {
unsafe {
(*info).flags = (*info).flags | 1;
}
irq = unsafe { __irq_domain_alloc_irqs(domain, irq, 1u32, node, info as *mut (), true, core::ptr::null::<irq_affinity_desc>()) };
if irq >= 0 {
irq_data = unsafe { irq_domain_get_irq_data(domain, irq as u32) };
let data: *mut mp_chip_data = (unsafe { (*irq_data).chip_data }) as *mut mp_chip_data;
unsafe {
(*data).isa_irq = true;
}
}
}
irq
}

fn mp_map_pin_to_irq(gsi: u32, idx: i32, ioapic: i32, pin: i32, flags: u32, info: *mut irq_alloc_info) -> i32 {
let domain: *mut irq_domain = mp_ioapic_irqdomain(ioapic);
let mut tmp: irq_alloc_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(tmp) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
let mut legacy: bool = false;
let mut irq: i32 = 0;
if domain.is_null() {
return 38i32.wrapping_neg();
}
let __v0: bool = idx >= 0 && test_bit__5!(unsafe { mp_irqs.0[idx as usize].srcbus }, unsafe { core::ptr::addr_of_mut!(mp_bus_not_pci.0) as *mut u64 }) != 0;
if __v0 {
irq = (unsafe { mp_irqs.0[idx as usize].srcbusirq }) as i32;
legacy = mp_is_legacy_irq(irq);
// IRQ2 is unusable for historical reasons on systems which
// have a legacy PIC. See the comment vs. IRQ2 further down.
//
// If this gets removed at some point then the related code
// in lapic_assign_system_vectors() needs to be adjusted as
// well.
if legacy && irq == 2 {
return 22i32.wrapping_neg();
}
}
let mut __UNIQUE_ID_guard_635: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_mutex) });
let __slate_cleanup_ptr_2 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_635) as *mut _;
let __slate_cleanup_2 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_2 as *mut _));
let mut __UNIQUE_ID_unlock_636: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!(ioapic_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_3 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_636) as *mut _;
let __slate_cleanup_3 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _));
if flags & 1 == 0 {
if !legacy {
irq = irq_find_mapping(domain, pin as u64) as i32;
if irq == 0 {
irq = 2i32.wrapping_neg();
}
}
} else {
ioapic_copy_alloc_attr(core::ptr::addr_of_mut!(tmp), info, gsi, ioapic, pin);
if legacy {
irq = alloc_isa_irq_from_domain(domain, irq, ioapic, pin, core::ptr::addr_of_mut!(tmp));
} else {
let __v4: i32 = irq_find_mapping(domain, pin as u64) as i32;
irq = __v4;
if __v4 == 0 {
irq = alloc_irq_from_domain(domain, ioapic, gsi, core::ptr::addr_of_mut!(tmp));
} else {
if !mp_check_pin_attr(irq, core::ptr::addr_of_mut!(tmp)) {
irq = 16i32.wrapping_neg();
}
}
}
if irq >= 0 {
let data: *mut mp_chip_data = irq_get_chip_data(irq as u32) as *mut mp_chip_data;
let __v6: u32 = unsafe { (*data).count };
unsafe {
(*data).count = __v6.wrapping_add(1);
}
}
}
irq
}

fn pin_2_irq(idx: i32, ioapic: i32, pin: i32, flags: u32) -> i32 {
let gsi: u32 = mp_pin_to_gsi(ioapic, pin);
// Debugging check, we are in big trouble if this message pops up!
if ((unsafe { mp_irqs.0[idx as usize].dstirq }) as i32) != pin {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013broken BIOS or MPTABLE parser, ayiee!!\n".as_ptr() as *const u8) };
}
mp_map_pin_to_irq(gsi, idx, ioapic, pin, flags, core::ptr::null_mut::<irq_alloc_info>())
}

#[unsafe(no_mangle)]
extern "C" fn mp_map_gsi_to_irq(gsi: u32, flags: u32, info: *mut irq_alloc_info) -> i32 {
let ioapic: i32 = mp_find_ioapic(gsi);
let mut pin: i32 = 0;
let mut idx: i32 = 0;
if ioapic < 0 {
return 19i32.wrapping_neg();
}
pin = mp_find_ioapic_pin(ioapic, gsi);
idx = find_irq_entry(ioapic, pin, 0i32);
if flags & 2 != 0 && idx < 0 {
return 19i32.wrapping_neg();
}
mp_map_pin_to_irq(gsi, idx, ioapic, pin, flags, info)
}

#[unsafe(no_mangle)]
extern "C" fn mp_unmap_irq(irq: i32) {
let irq_data: *mut irq_data = unsafe { irq_get_irq_data(irq as u32) };
if irq_data.is_null() || unsafe { (*irq_data).domain }.is_null() {
return;
}
let data: *mut mp_chip_data = (unsafe { (*irq_data).chip_data }) as *mut mp_chip_data;
if data.is_null() || unsafe { (*data).isa_irq } {
return;
}
let mut __UNIQUE_ID_guard_637: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_637) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_638: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!(ioapic_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_638) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
let __v3: u32 = unsafe { (*data).count };
let __v4: u32 = __v3.wrapping_sub(1);
unsafe {
(*data).count = __v4;
}
if __v4 == 0 {
unsafe { irq_domain_free_irqs(irq as u32, 1u32) };
}
}

/// Find a specific PCI IRQ entry.
/// Not an __init, possibly needed by modules
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.io_apic.IO_APIC_get_PCI_irq_vector")]
extern "C" fn IO_APIC_get_PCI_irq_vector(bus: i32, slot: i32, pin: i32) -> i32 {
let mut found: i32 = unsafe { core::mem::zeroed() };
let mut ioapic_idx: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_5: *mut i32 = core::ptr::addr_of_mut!(ioapic_idx) as *mut i32;
let mut best_idx: i32 = unsafe { core::mem::zeroed() };
let mut best_ioapic: i32 = unsafe { core::mem::zeroed() };
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut irq: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut i32 = core::ptr::addr_of_mut!(irq) as *mut i32;
unsafe {
let __slate_auto_bytes_1 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
best_ioapic = 1i32.wrapping_neg();
best_idx = 1i32.wrapping_neg();
loop {
if 2 <= apic_verbosity {
while false {
}
_printk(c"\x017Querying PCI -> IRQ mapping bus:%d, slot:%d, pin:%d.\n".as_ptr() as *const u8, bus, slot, pin);
}
if true {
break;
}
}
if test_bit!(bus, core::ptr::addr_of_mut!(mp_bus_not_pci.0) as *mut u64) != 0 {
loop {
if 1 <= apic_verbosity {
while false {
}
_printk(c"\x016PCI BIOS passed nonexistent PCI bus %d!\n".as_ptr() as *const u8, bus);
}
if true {
break;
}
}
return 1i32.wrapping_neg();
} else {
*__slate_slot_11 = 0;
'__join_0: {
'__join_7: {
while *__slate_slot_11 < mp_irq_entries {
let lbus: i32 = mp_irqs.0[*__slate_slot_11 as usize].srcbus as i32;
let __slate_auto_bytes = __slate_slot_5 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
found = 0;
if !(bus != lbus || (mp_irqs.0[*__slate_slot_11 as usize].irqtype as i32) != 0 || slot != (mp_irqs.0[*__slate_slot_11 as usize].srcbusirq as i32) >> 2i32 & 31) {
*__slate_slot_5 = 0;
'__join_10: {
loop {
if *__slate_slot_5 < nr_ioapics {
if mpc_ioapic_id(*__slate_slot_5) == (mp_irqs.0[*__slate_slot_11 as usize].dstapic as i32) || (mp_irqs.0[*__slate_slot_11 as usize].dstapic as i32) == 255 {
break;
} else {
let __v3: i32 = *__slate_slot_5;
*__slate_slot_5 = __v3.wrapping_add(1);
}
} else {
break '__join_10;
}
}
found = 1;
}
if found != 0 {
// Skip ISA IRQs
*__slate_slot_12 = pin_2_irq(*__slate_slot_11, *__slate_slot_5, mp_irqs.0[*__slate_slot_11 as usize].dstirq as i32, 0u32);
if !(*__slate_slot_12 > 0 && !IO_APIC_IRQ__2!(*__slate_slot_12)) {
if pin == (mp_irqs.0[*__slate_slot_11 as usize].srcbusirq as i32) & 3 {
break '__join_7;
} else {
// Use the first all-but-pin matching entry as a
// best-guess fuzzy result for broken mptables.
if best_idx < 0 {
best_idx = *__slate_slot_11;
best_ioapic = *__slate_slot_5;
}
}
}
}
}
let __v1: i32 = *__slate_slot_11;
*__slate_slot_11 = __v1.wrapping_add(1);
}
if best_idx < 0 {
return 1i32.wrapping_neg();
} else {
break '__join_0;
}
}
best_idx = *__slate_slot_11;
best_ioapic = *__slate_slot_5;
}
return pin_2_irq(best_idx, best_ioapic, mp_irqs.0[best_idx as usize].dstirq as i32, 1u32);
}
}
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_IO_APIC_get_PCI_irq_vector_639: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn(i32, i32, i32) -> i32>, *mut ()>(Some(IO_APIC_get_PCI_irq_vector)) };

#[unsafe(link_section = ".data..read_mostly")]
static mut ioapic_chip: irq_chip = {
let mut __t0: irq_chip = unsafe { core::mem::zeroed() };
__t0.name = (b"IO-APIC\0".as_ptr() as *mut u8) as *const u8;
__t0.irq_startup = Some(startup_ioapic_irq);
__t0.irq_ack = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_data)>>(irq_chip_ack_parent as *const ()) };
__t0.irq_mask = Some(mask_ioapic_irq);
__t0.irq_unmask = Some(unmask_ioapic_irq);
__t0.irq_eoi = Some(ioapic_ack_level);
__t0.irq_set_affinity = Some(ioapic_set_affinity);
__t0.irq_retrigger = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_data) -> i32>>(irq_chip_retrigger_hierarchy as *const ()) };
__t0.irq_get_irqchip_state = Some(ioapic_irq_get_chip_state);
__t0.flags = 5136u64;
__t0
};

#[unsafe(link_section = ".data..read_mostly")]
static mut ioapic_ir_chip: irq_chip = {
let mut __t0: irq_chip = unsafe { core::mem::zeroed() };
__t0.name = (b"IR-IO-APIC\0".as_ptr() as *mut u8) as *const u8;
__t0.irq_startup = Some(startup_ioapic_irq);
__t0.irq_ack = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_data)>>(irq_chip_ack_parent as *const ()) };
__t0.irq_mask = Some(mask_ioapic_irq);
__t0.irq_unmask = Some(unmask_ioapic_irq);
__t0.irq_eoi = Some(ioapic_ir_ack_level);
__t0.irq_set_affinity = Some(ioapic_set_affinity);
__t0.irq_retrigger = unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_data) -> i32>>(irq_chip_retrigger_hierarchy as *const ()) };
__t0.irq_get_irqchip_state = Some(ioapic_irq_get_chip_state);
__t0.flags = 1040u64;
__t0
};

#[unsafe(link_section = ".init.text")]
fn setup_IO_APIC_irqs() {
let mut ioapic: u32 = 0;
let mut pin: u32 = 0;
let mut idx: i32 = 0;
loop {
if 1 <= unsafe { apic_verbosity } {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016Init IO_APIC IRQs\n".as_ptr() as *const u8) };
}
if true {
break;
}
}
while ioapic < ((unsafe { nr_ioapics }) as u32) {
pin = 0;
while pin < ((unsafe { ioapics.0[ioapic as usize].nr_registers }) as u32) {
idx = find_irq_entry(ioapic as i32, pin as i32, 0i32);
if idx < 0 {
loop {
if 1 <= unsafe { apic_verbosity } {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x016apic %d pin %d not connected\n".as_ptr() as *const u8, mpc_ioapic_id(ioapic as i32), pin) };
}
if true {
break;
}
}
} else {
pin_2_irq(idx, ioapic as i32, pin as i32, (if ioapic != 0 { 0i32 } else { 1i32 }) as u32);
}
pin = pin.wrapping_add(1);
}
ioapic = ioapic.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn ioapic_zap_locks() {
let mut __slate_compound_0: raw_spinlock = unsafe { core::mem::zeroed() };
'__slate_break_1: loop {
unsafe {
*core::ptr::addr_of_mut!(ioapic_lock) = *{
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
}

fn io_apic_print_entries(apic__0: u32, nr_entries: u32) {
let mut entry: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut buf: __SlateAlign16<[u8; 256]> = __SlateAlign16([0u8; 256]);
let mut i: i32 = 0;
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x017IOAPIC %d:\n".as_ptr() as *const u8, apic__0) };
while (i as u32) <= nr_entries {
entry = ioapic_read_entry(apic__0 as i32, i);
unsafe { snprintf(buf.0.as_mut_ptr() as *mut u8, core::mem::size_of::<[u8; 256]>() as u64, c" pin%02x, %s, %s, %s, V(%02X), IRR(%1d), S(%1d)".as_ptr() as *const u8, i, if (core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_masked() as i32) != 0 { c"disabled".as_ptr() as *mut u8 } else { c"enabled ".as_ptr() as *mut u8 }, if (core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_is_level() as i32) != 0 { c"level".as_ptr() as *mut u8 } else { c"edge ".as_ptr() as *mut u8 }, if (core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_active_low() as i32) != 0 { c"low ".as_ptr() as *mut u8 } else { c"high".as_ptr() as *mut u8 }, core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_vector() as i32, core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_irr() as i32, core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_delivery_status() as i32) };
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_1.__slate_bits_0)).__get_ir_format() }) as i32) != 0 {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x017%s, remapped, I(%04X),  Z(%X)\n".as_ptr() as *const u8, buf.0.as_mut_ptr() as *mut u8, (core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_1.__slate_bits_0)).__get_ir_index_15() as i32) << 15i32 | (core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_1.__slate_bits_0)).__get_ir_index_0_14() as i32), core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_1.__slate_bits_0)).__get_ir_zero() as i32) };
} else {
let mut __v4: i32;
while false {
}
__v4 = unsafe { _printk(c"\x017%s, %s, D(%02X%02X), M(%1d)\n".as_ptr() as *const u8, buf.0.as_mut_ptr() as *mut u8, if (core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_dest_mode_logical() as i32) != 0 { c"logical ".as_ptr() as *mut u8 } else { c"physical".as_ptr() as *mut u8 }, core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_virt_destid_8_14() as i32, core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_destid_0_7() as i32, core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_delivery_mode() as i32) };
}
i = i.wrapping_add(1);
}
}

#[unsafe(link_section = ".init.text")]
fn print_IO_APIC(ioapic_idx: i32) {
let mut __UNIQUE_ID_unlock_641: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_14: *mut *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_641) as *mut *mut __SlateRecord_spinlock_h_594;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_15: *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_594;
let mut reg_03: core::mem::MaybeUninit<IO_APIC_reg_03> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut IO_APIC_reg_03 = core::ptr::addr_of_mut!(reg_03) as *mut IO_APIC_reg_03;
let mut reg_02: core::mem::MaybeUninit<IO_APIC_reg_02> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut IO_APIC_reg_02 = core::ptr::addr_of_mut!(reg_02) as *mut IO_APIC_reg_02;
let mut reg_01: core::mem::MaybeUninit<IO_APIC_reg_01> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut IO_APIC_reg_01 = core::ptr::addr_of_mut!(reg_01) as *mut IO_APIC_reg_01;
let mut reg_00: core::mem::MaybeUninit<IO_APIC_reg_00> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut IO_APIC_reg_00 = core::ptr::addr_of_mut!(reg_00) as *mut IO_APIC_reg_00;
let __slate_cleanup_ptr_15 = __slate_slot_15;
let __slate_cleanup_armed_15 = core::cell::Cell::new(false);
let __slate_cleanup_15 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_15.get() {
unsafe { class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_15 as *mut _) };
}
});
let __slate_cleanup_ptr_14 = __slate_slot_14;
let __slate_cleanup_armed_14 = core::cell::Cell::new(false);
let __slate_cleanup_14 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_14.get() {
unsafe { __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_14 as *mut _) };
}
});
unsafe {
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
core::ptr::write(__slate_slot_15, class_raw_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(ioapic_lock)));
__slate_cleanup_armed_15.set(true);
core::ptr::write(__slate_slot_14, (core::ptr::addr_of_mut!(ioapic_lock) as u64) as *mut __SlateRecord_spinlock_h_594);
__slate_cleanup_armed_14.set(true);
if !class_raw_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_15))).is_null() || !class_raw_spinlock_irqsave_is_conditional {
if true {
(*__slate_slot_19).raw = io_apic_read(ioapic_idx as u32, 0u32);
(*__slate_slot_18).raw = io_apic_read(ioapic_idx as u32, 1u32);
if ((*__slate_slot_18).bits.__slate_bits_0.__get_version() as i32) >= 16 {
(*__slate_slot_17).raw = io_apic_read(ioapic_idx as u32, 2u32);
}
if ((*__slate_slot_18).bits.__slate_bits_0.__get_version() as i32) >= 32 {
(*__slate_slot_16).raw = io_apic_read(ioapic_idx as u32, 3u32);
}
}
}
__slate_cleanup_armed_14.set(false);
__class_raw_spinlock_irqsave_cleanup_ctx(__slate_slot_14 as *mut _);
__slate_cleanup_armed_15.set(false);
class_raw_spinlock_irqsave_destructor(__slate_slot_15 as *mut _);
while false {
}
_printk(c"\x017IO APIC #%d......\n".as_ptr() as *const u8, mpc_ioapic_id(ioapic_idx));
while false {
}
_printk(c"\x017.... register #00: %08X\n".as_ptr() as *const u8, (*__slate_slot_19).raw);
while false {
}
_printk(c"\x017.......    : physical APIC id: %02X\n".as_ptr() as *const u8, (*__slate_slot_19).bits.__slate_bits_0.__get_ID() as i32);
while false {
}
_printk(c"\x017.......    : Delivery Type: %X\n".as_ptr() as *const u8, (*__slate_slot_19).bits.__slate_bits_0.__get_delivery_type() as i32);
while false {
}
_printk(c"\x017.......    : LTS          : %X\n".as_ptr() as *const u8, (*__slate_slot_19).bits.__slate_bits_0.__get_LTS() as i32);
while false {
}
_printk(c"\x017.... register #01: %08X\n".as_ptr() as *const u8, *(core::ptr::addr_of_mut!(*__slate_slot_18) as *mut i32));
while false {
}
_printk(c"\x017.......     : max redirection entries: %02X\n".as_ptr() as *const u8, (*__slate_slot_18).bits.__slate_bits_0.__get_entries() as i32);
while false {
}
_printk(c"\x017.......     : PRQ implemented: %X\n".as_ptr() as *const u8, (*__slate_slot_18).bits.__slate_bits_0.__get_PRQ() as i32);
while false {
}
_printk(c"\x017.......     : IO APIC version: %02X\n".as_ptr() as *const u8, (*__slate_slot_18).bits.__slate_bits_0.__get_version() as i32);
// Some Intel chipsets with IO APIC VERSION of 0x1? don't have reg_02,
// but the value of reg_02 is read as the previous read register
// value, so ignore it if reg_02 == reg_01.
if ((*__slate_slot_18).bits.__slate_bits_0.__get_version() as i32) >= 16 && (*__slate_slot_17).raw != (*__slate_slot_18).raw {
while false {
}
_printk(c"\x017.... register #02: %08X\n".as_ptr() as *const u8, (*__slate_slot_17).raw);
while false {
}
_printk(c"\x017.......     : arbitration: %02X\n".as_ptr() as *const u8, (*__slate_slot_17).bits.__slate_bits_0.__get_arbitration() as i32);
}
// Some Intel chipsets with IO APIC VERSION of 0x2? don't have reg_02
// or reg_03, but the value of reg_0[23] is read as the previous read
// register value, so ignore it if reg_03 == reg_0[12].
if ((*__slate_slot_18).bits.__slate_bits_0.__get_version() as i32) >= 32 && (*__slate_slot_16).raw != (*__slate_slot_17).raw && (*__slate_slot_16).raw != (*__slate_slot_18).raw {
while false {
}
_printk(c"\x017.... register #03: %08X\n".as_ptr() as *const u8, (*__slate_slot_16).raw);
while false {
}
_printk(c"\x017.......     : Boot DT    : %X\n".as_ptr() as *const u8, (*__slate_slot_16).bits.__slate_bits_0.__get_boot_DT() as i32);
}
while false {
}
_printk(c"\x017.... IRQ redirection table:\n".as_ptr() as *const u8);
io_apic_print_entries(ioapic_idx as u32, ((*__slate_slot_18).bits.__slate_bits_0.__get_entries() as i32) as u32);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn print_IO_APICs() {
let mut ioapic_idx: i32 = 0;
let mut irq: u32 = 0;
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x017number of MP IRQ sources: %d.\n".as_ptr() as *const u8, mp_irq_entries) };
while ioapic_idx < unsafe { nr_ioapics } {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x017number of IO-APIC #%d registers: %d.\n".as_ptr() as *const u8, mpc_ioapic_id(ioapic_idx), ioapics.0[ioapic_idx as usize].nr_registers) };
ioapic_idx = ioapic_idx.wrapping_add(1);
}
// We are a bit conservative about what we expect.  We have to
// know about every hardware change ASAP.
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x016testing the IO APIC.......................\n".as_ptr() as *const u8) };
ioapic_idx = 0;
while ioapic_idx < unsafe { nr_ioapics } {
print_IO_APIC(ioapic_idx);
ioapic_idx = ioapic_idx.wrapping_add(1);
}
let mut __v12: i32;
while false {
}
__v12 = unsafe { _printk(c"\x017IRQ to pin mappings:\n".as_ptr() as *const u8) };
let mut __nr_irqs__: u32 = unsafe { irq_get_nr_irqs() };
while __nr_irqs__ != 0 {
irq = unsafe { irq_get_next_irq(0u32) };
while irq < __nr_irqs__ {
let mut entry: *mut irq_pin_list = unsafe { core::mem::zeroed() };
let mut data: *mut mp_chip_data = unsafe { core::mem::zeroed() };
let chip: *mut irq_chip = irq_get_chip(irq);
if !(chip != unsafe { core::ptr::addr_of_mut!(ioapic_chip) } && chip != unsafe { core::ptr::addr_of_mut!(ioapic_ir_chip) }) {
data = irq_get_chip_data(irq) as *mut mp_chip_data;
if !data.is_null() {
if list_empty((unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }) as *const list_head) == 0 {
let mut __v15: i32;
while false {
}
__v15 = unsafe { _printk(c"\x017IRQ%d ".as_ptr() as *const u8, irq) };
entry = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) });
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }, list) == 0 {
let mut __v18: i32;
while false {
}
__v18 = unsafe { _printk(c"\x01c-> %d:%d".as_ptr() as *const u8, (*entry).apic, (*entry).pin) };
entry = list_next_entry__2!(entry, list);
}
let mut __v22: i32;
while false {
}
__v22 = unsafe { _printk(c"\x01c\n".as_ptr() as *const u8) };
}
}
}
irq = unsafe { irq_get_next_irq(irq.wrapping_add(1)) };
}
__nr_irqs__ = 0;
}
let mut __v25: i32;
while false {
}
__v25 = unsafe { _printk(c"\x016.................................... done.\n".as_ptr() as *const u8) };
}

/// Where if anywhere is the i8259 connect in external int mode
#[repr(C)]
#[derive(Clone, Copy)]
struct __SlateRecord_io_apic_c_1270 {
    pin: i32,
    apic: i32,
}


static mut ioapic_i8259: __SlateRecord_io_apic_c_1270 = __SlateRecord_io_apic_c_1270 { pin: 1i32.wrapping_neg(), apic: 1i32.wrapping_neg() };

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn enable_IO_APIC() {
let mut i8259_apic: i32 = 0;
let mut i8259_pin: i32 = 0;
let mut apic__0: i32 = 0;
let mut pin: i32 = 0;
if unsafe { ioapic_is_disabled } {
unsafe {
nr_ioapics = 0;
}
}
if nr_legacy_irqs() == 0 || (unsafe { nr_ioapics }) == 0 {
return;
}
while apic__0 < unsafe { nr_ioapics } {
pin = 0;
while pin < unsafe { ioapics.0[apic__0 as usize].nr_registers } {
// See if any of the pins is in ExtINT mode
let entry: IO_APIC_route_entry = ioapic_read_entry(apic__0, pin);
// If the interrupt line is enabled and in ExtInt mode I
// have found the pin where the i8259 is connected.
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_masked() }) as i32) == 0 && ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_delivery_mode() }) as i32) == 7 {
unsafe {
ioapic_i8259.apic = apic__0;
}
unsafe {
ioapic_i8259.pin = pin;
}
break;
}
pin = pin.wrapping_add(1);
}
apic__0 = apic__0.wrapping_add(1);
}
// Look to see what if the MP table has reported the ExtINT
//
// If we could not find the appropriate pin by looking at the ioapic
// the i8259 probably is not connected the ioapic but give the
// mptable a chance anyway.
i8259_pin = find_isa_irq_pin(0i32, 3i32);
i8259_apic = find_isa_irq_apic(0i32, 3i32);
// Trust the MP table if nothing is setup in the hardware
if (unsafe { ioapic_i8259.pin }) == 1i32.wrapping_neg() && i8259_pin >= 0 {
let mut __v6: i32;
while false {
}
__v6 = unsafe { _printk(c"\x014ExtINT not setup in hardware but reported by MP table\n".as_ptr() as *const u8) };
unsafe {
ioapic_i8259.pin = i8259_pin;
}
unsafe {
ioapic_i8259.apic = i8259_apic;
}
}
// Complain if the MP table and the hardware disagree
if ((unsafe { ioapic_i8259.apic }) != i8259_apic || (unsafe { ioapic_i8259.pin }) != i8259_pin) && i8259_pin >= 0 && (unsafe { ioapic_i8259.pin }) >= 0 {
let mut __v8: i32;
while false {
}
__v8 = unsafe { _printk(c"\x014ExtINT in hardware and MP table differ\n".as_ptr() as *const u8) };
}
// Do not trust the IO-APIC being empty at bootup
clear_IO_APIC();
}

#[unsafe(no_mangle)]
extern "C" fn native_restore_boot_irq_mode() {
// If the i8259 is routed through an IOAPIC Put that IOAPIC in
// virtual wire mode so legacy interrupts can be delivered.
if (unsafe { ioapic_i8259.pin }) != 1i32.wrapping_neg() {
let mut entry: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let apic_id: u32 = read_apic_id();
unsafe { memset(core::ptr::addr_of_mut!(entry) as *mut (), 0i32, core::mem::size_of::<IO_APIC_route_entry>() as u64) };
let __t0 = 0u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t1.__set_masked(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t1);
}
let __t2 = 0u64;
unsafe {
let mut __t3 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t3.__set_is_level(__t2);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t3);
}
let __t4 = 0u64;
unsafe {
let mut __t5 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t5.__set_active_low(__t4);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t5);
}
let __t6 = 0u64;
unsafe {
let mut __t7 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t7.__set_dest_mode_logical(__t6);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t7);
}
let __t8 = 7u64;
unsafe {
let mut __t9 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t9.__set_delivery_mode(__t8);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t9);
}
let __t10 = (apic_id & 255) as u64;
unsafe {
let mut __t11 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t11.__set_destid_0_7(__t10);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t11);
}
let __t12 = (apic_id >> 8i32) as u64;
unsafe {
let mut __t13 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t13.__set_virt_destid_8_14(__t12);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t13);
}
// Add it to the IO-APIC irq-routing table
ioapic_write_entry(unsafe { ioapic_i8259.apic }, unsafe { ioapic_i8259.pin }, entry);
}
let __v0: bool = boot_cpu_has!(9i32) != 0 || apic_from_smp_config();
if __v0 {
unsafe { disconnect_bsp_APIC((ioapic_i8259.pin != 1i32.wrapping_neg()) as i32) };
}
}

#[unsafe(no_mangle)]
extern "C" fn restore_boot_irq_mode() {
if nr_legacy_irqs() == 0 {
return;
}
unsafe { unsafe { x86_apic_ops.restore }.unwrap()() };
}

#[unsafe(link_section = ".init.text")]
extern "C" fn notimercheck(s: *mut u8) -> i32 {
unsafe {
no_timer_check = 1;
}
1
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_notimercheck: [u8; 15] = [110, 111, 95, 116, 105, 109, 101, 114, 95, 99, 104, 101, 99, 107, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_notimercheck: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_notimercheck) as *const u8 };
__t0.setup_func = Some(notimercheck);
__t0.early = 0i32;
__t0
};

#[unsafe(link_section = ".init.text")]
fn delay_with_tsc() {
let mut start: u64 = 0;
let mut now: u64 = 0;
let end: u64 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_add(4);
start = rdtsc();
// We don't know the TSC frequency yet, but waiting for
// 40000000000/HZ TSC cycles is safe:
// 4 GHz == 10 jiffies
// 1 GHz == 40 jiffies
loop {
native_pause();
now = rdtsc();
let __v1: bool = now.wrapping_sub(start) < 40000000000 / 1000 && time_before_eq!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, end);
if !__v1 {
break;
}
}
}

#[unsafe(link_section = ".init.text")]
fn delay_without_tsc() {
let end: u64 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }.wrapping_add(4);
let mut band: i32 = 1;
// We don't know any frequency yet, but waiting for
// 40940000000/HZ cycles is safe:
// 4 GHz == 10 jiffies
// 1 GHz == 40 jiffies
// 1 << 1 + 1 << 2 +...+ 1 << 11 = 4094
loop {
let __v0: i32 = band;
band = __v0.wrapping_add(1);
unsafe { __delay(((1u32 << __v0) as u64).wrapping_mul(10000000) / 1000) };
let __v3: bool = band < 12 && time_before_eq!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, end);
if !__v3 {
break;
}
}
}

/// There is a nasty bug in some older SMP boards, their mptable lies
/// about the timer IRQ. We do the following to work around the situation:
///
/// 	- timer IRQ defaults to IO-APIC IRQ
/// 	- if this function detects that timer IRQs are defunct, then we fall
/// 	  back to ISA timer IRQs
#[unsafe(link_section = ".init.text")]
fn timer_irq_works() -> i32 {
let t1: u64 = unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) };
if (unsafe { no_timer_check }) != 0 {
return 1;
}
local_irq_enable!();
if boot_cpu_has!(4i32) != 0 {
delay_with_tsc();
} else {
delay_without_tsc();
}
// Expect a few ticks at least, to be sure some possible
// glue logic does not lock up after one or two first
// ticks in a non-ExtINT mode.  Also the local APIC
// might have cached one ExtINT interrupt.  Finally, at
// least one tick may be lost due to delays.
local_irq_disable!();
// Did jiffies advance?
time_after__2!(unsafe { core::ptr::read_volatile(core::ptr::addr_of!(jiffies)) }, t1.wrapping_add(4)) as i32
}

/// In the SMP+IOAPIC case it might happen that there are an unspecified
/// number of pending IRQ events unhandled. These cases are very rare,
/// so we 'resend' these IRQs via IPIs, to the same CPU. It's much
/// better to do it this way as thus we do not have to be aware of
/// 'pending' interrupts in the IRQ path, except at this point.
///
///
/// Edge triggered needs to resend any interrupt that was delayed but this
/// is now handled in the device independent code.
///
/// Starting up a edge-triggered IO-APIC interrupt is nasty - we need to
/// make sure that we get the edge.  If it is already asserted for some
/// reason, we need return 1 to indicate that is was pending.
///
/// This is not complete - we should be able to fake an edge even if it
/// isn't on the 8259A...
#[unsafe(link_section = ".text.slate_distinct.io_apic.startup_ioapic_irq")]
extern "C" fn startup_ioapic_irq(data: *mut irq_data) -> u32 {
let mut was_pending: i32 = 0;
let irq: i32 = (unsafe { (*data).irq }) as i32;
let mut __UNIQUE_ID_guard_642: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_642) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_643: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_643) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if irq < nr_legacy_irqs() {
unsafe { unsafe { (*legacy_pic).mask }.unwrap()(irq as u32) };
if (unsafe { unsafe { (*legacy_pic).irq_pending }.unwrap()(irq as u32) }) != 0 {
was_pending = 1;
}
}
__unmask_ioapic((unsafe { (*data).chip_data }) as *mut mp_chip_data);
was_pending as u32
}

fn io_apic_level_ack_pending(data: *mut mp_chip_data) -> bool {
let mut entry: *mut irq_pin_list = unsafe { core::mem::zeroed() };
let mut __UNIQUE_ID_guard_644: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_644) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_645: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_645) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
entry = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) });
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }, list) == 0 {
let mut e: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(e) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let pin: i32 = unsafe { (*entry).pin };
let __t0 = io_apic_read((unsafe { (*entry).apic }) as u32, 16i32.wrapping_add(pin.wrapping_mul(2)) as u32) as u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(e.__slate_anon_0.__slate_anon_2.__slate_bits_0));
__t1.__set_w1(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(e.__slate_anon_0.__slate_anon_2.__slate_bits_0), __t1);
}
// Is the remote IRR bit set?
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(e.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_irr() }) as i32) != 0 {
return true;
}
entry = list_next_entry__2!(entry, list);
}
false
}

fn ioapic_prepare_move(data: *mut irq_data) -> bool {
// If we are moving the IRQ we need to mask it
if unlikely__2!(irqd_is_setaffinity_pending(data)) != 0 {
if !irqd_irq_masked(data) {
mask_ioapic_irq(data);
}
return true;
}
false
}

fn ioapic_finish_move(data: *mut irq_data, moveit: bool) {
if unlikely__2!(moveit) != 0 {
// Only migrate the irq if the ack has been received.
//
// On rare occasions the broadcast level triggered ack gets
// delayed going to ioapics, and if we reprogram the
// vector while Remote IRR is still set the irq will never
// fire again.
//
// To prevent this scenario we read the Remote IRR bit
// of the ioapic.  This has two effects.
// - On any sane system the read of the ioapic will
//   flush writes (and acks) going to the ioapic from
//   this cpu.
// - We get to see if the ACK has actually been delivered.
//
// Based on failed experiments of reprogramming the
// ioapic entry from outside of irq context starting
// with masking the ioapic entry and then polling until
// Remote IRR was clear before reprogramming the
// ioapic I don't trust the Remote IRR bit to be
// completely accurate.
//
// However there appears to be no other way to plug
// this race, so if the Remote IRR bit is not
// accurate and is causing problems then it is a hardware bug
// and you can go talk to the chipset vendor about it.
if !io_apic_level_ack_pending((unsafe { (*data).chip_data }) as *mut mp_chip_data) {
unsafe { irq_move_masked_irq(data) };
}
// If the IRQ is masked in the core, leave it:
if !irqd_irq_masked(data) {
unmask_ioapic_irq(data);
}
}
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.ioapic_ack_level")]
extern "C" fn ioapic_ack_level(irq_data: *mut irq_data) {
let cfg: *mut irq_cfg = unsafe { irqd_cfg(irq_data) };
let mut v: u64 = 0;
let mut moveit: bool = false;
let mut i: i32 = 0;
unsafe { irq_complete_move(cfg) };
moveit = ioapic_prepare_move(irq_data);
// It appears there is an erratum which affects at least version 0x11
// of I/O APIC (that's the 82093AA and cores integrated into various
// chipsets).  Under certain conditions a level-triggered interrupt is
// erroneously delivered as edge-triggered one but the respective IRR
// bit gets set nevertheless.  As a result the I/O unit expects an EOI
// message but it will never arrive and further interrupts are blocked
// from the source.  The exact reason is so far unknown, but the
// phenomenon was observed when two consecutive interrupt requests
// from a given source get delivered to the same CPU and the source is
// temporarily disabled in between.
//
// A workaround is to simulate an EOI message manually.  We achieve it
// by setting the trigger mode to edge and then to level when the edge
// trigger mode gets detected in the TMR of a local APIC for a
// level-triggered interrupt.  We mask the source for the time of the
// operation to prevent an edge-triggered interrupt escaping meanwhile.
// The idea is from Manfred Spraul.  --macro
//
// Also in the case when cpu goes offline, fixup_irqs() will forward
// any unhandled interrupt on the offlined cpu to the new cpu
// destination that is handling the corresponding interrupt. This
// interrupt forwarding is done via IPI's. Hence, in this case also
// level-triggered io-apic interrupt will be seen as an edge
// interrupt in the IRR. And we can't rely on the cpu's EOI
// to be broadcasted to the IO-APIC's which will clear the remoteIRR
// corresponding to the level-triggered interrupt. Hence on IO-APIC's
// supporting EOI register, we do an explicit EOI to clear the
// remote IRR and on IO-APIC's which don't have an EOI register,
// we use the above logic (mask+edge followed by unmask+level) from
// Manfred Spraul to clear the remote IRR.
i = (unsafe { (*cfg).vector }) as i32;
v = apic_read(384i32.wrapping_add((i & !31) >> 1i32) as u32) as u64;
// We must acknowledge the irq before we move it or the acknowledge will
// not propagate properly.
apic_eoi();
// Tail end of clearing remote IRR bit (either by delivering the EOI
// message via io-apic EOI register write or simulating it using
// mask+edge followed by unmask+level logic) manually when the
// level triggered interrupt is seen as the edge triggered interrupt
// at the cpu.
if v & ((1i32 << (i & 31)) as u64) == 0 {
unsafe { irq_stat_inc_and_enable(17u32) };
eoi_ioapic_pin((unsafe { (*cfg).vector }) as i32, (unsafe { (*irq_data).chip_data }) as *mut mp_chip_data);
}
ioapic_finish_move(irq_data, moveit);
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.ioapic_ir_ack_level")]
extern "C" fn ioapic_ir_ack_level(irq_data: *mut irq_data) {
let data: *mut mp_chip_data = (unsafe { (*irq_data).chip_data }) as *mut mp_chip_data;
// Intr-remapping uses pin number as the virtual vector
// in the RTE. Actual vector is programmed in
// intr-remapping table entry. Hence for the io-apic
// EOI we use the pin number.
unsafe { apic_ack_irq(irq_data) };
eoi_ioapic_pin((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!((*data).entry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_vector() }) as i32, data);
}

/// The I/OAPIC is just a device for generating MSI messages from legacy
/// interrupt pins. Various fields of the RTE translate into bits of the
/// resulting MSI which had a historical meaning.
///
/// With interrupt remapping, many of those bits have different meanings
/// in the underlying MSI, but the way that the I/OAPIC transforms them
/// from its RTE to the MSI message is the same. This function allows
/// the parent IRQ domain to compose the MSI message, then takes the
/// relevant bits to put them in the appropriate places in the RTE in
/// order to generate that message when the IRQ happens.
///
/// The setup here relies on a preconfigured route entry (is_level,
/// active_low, masked) because the parent domain is merely composing the
/// generic message routing information which is used for the MSI.
fn ioapic_setup_msg_from_msi(irq_data: *mut irq_data, entry: *mut IO_APIC_route_entry) {
let mut msg: msi_msg = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(msg) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 12usize);
}
// Let the parent domain compose the MSI message
unsafe { irq_chip_compose_msi_msg(irq_data, core::ptr::addr_of_mut!(msg)) };
// - Real vector
// - DMAR/IR: 8bit subhandle (ioapic.pin)
// - AMD/IR:  8bit IRTE index
let __t1 = ((unsafe {
let __t0 = (core::ptr::addr_of!(msg.__slate_anon_2.arch_data.__slate_anon_0) as *const __SlateRecord_msi_h_26).read_unaligned();
__t0.__slate_anon_0.__slate_bits_0.__get_vector()
}) as i32) as u64;
unsafe {
let mut __t2 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t2.__set_vector(__t1);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0), __t2);
}
// Delivery mode (for DMAR/IR all 0)
let __t4 = ((unsafe {
let __t3 = (core::ptr::addr_of!(msg.__slate_anon_2.arch_data.__slate_anon_0) as *const __SlateRecord_msi_h_26).read_unaligned();
__t3.__slate_anon_0.__slate_bits_0.__get_delivery_mode()
}) as i32) as u64;
unsafe {
let mut __t5 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t5.__set_delivery_mode(__t4);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0), __t5);
}
// Destination mode or DMAR/IR index bit 15
let __t6 = ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(msg.__slate_anon_0.arch_addr_lo.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_dest_mode_logical() }) as i32) as u64;
unsafe {
let mut __t7 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t7.__set_dest_mode_logical(__t6);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0), __t7);
}
// DMAR/IR: 1, 0 for all other modes
let __t8 = ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(msg.__slate_anon_0.arch_addr_lo.__slate_anon_0.__slate_anon_1.__slate_bits_0)).__get_dmar_format() }) as i32) as u64;
unsafe {
let mut __t9 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_1.__slate_bits_0));
__t9.__set_ir_format(__t8);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_1.__slate_bits_0), __t9);
}
// - DMAR/IR: index bit 0-14.
//
// - Virt: If the host supports x2apic without a virtualized IR
// 	   unit then bit 0-6 of dmar_index_0_14 are providing bit
// 	   8-14 of the destination id.
//
// All other modes have bit 0-6 of dmar_index_0_14 cleared and the
// topmost 8 bits are destination id bit 0-7 (entry::destid_0_7).
let __t10 = ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(msg.__slate_anon_0.arch_addr_lo.__slate_anon_0.__slate_anon_1.__slate_bits_0)).__get_dmar_index_0_14() }) as i32) as u64;
unsafe {
let mut __t11 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_1.__slate_bits_0));
__t11.__set_ir_index_0_14(__t10);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_1.__slate_bits_0), __t11);
}
}

fn ioapic_configure_entry(irqd: *mut irq_data) {
let mpd: *mut mp_chip_data = (unsafe { (*irqd).chip_data }) as *mut mp_chip_data;
let mut entry: *mut irq_pin_list = unsafe { core::mem::zeroed() };
ioapic_setup_msg_from_msi(irqd, unsafe { core::ptr::addr_of_mut!((*mpd).entry) });
entry = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*mpd).irq_2_pin) });
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*mpd).irq_2_pin) }, list) == 0 {
__ioapic_write_entry(unsafe { (*entry).apic }, unsafe { (*entry).pin }, unsafe { (*mpd).entry });
entry = list_next_entry__2!(entry, list);
}
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.ioapic_set_affinity")]
extern "C" fn ioapic_set_affinity(irq_data: *mut irq_data, mask: *const cpumask, force: bool) -> i32 {
let parent: *mut irq_data = unsafe { (*irq_data).parent_data };
let ret: i32 = unsafe { unsafe { (*(*parent).chip).irq_set_affinity }.unwrap()(parent, mask, force) };
let mut __UNIQUE_ID_guard_646: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_646) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_647: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_647) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
if ret >= 0 && ret != 2 {
ioapic_configure_entry(irq_data);
}
ret
}

/// Interrupt shutdown masks the ioapic pin, but the interrupt might already
/// be in flight, but not yet serviced by the target CPU. That means
/// __synchronize_hardirq() would return and claim that everything is calmed
/// down. So free_irq() would proceed and deactivate the interrupt and free
/// resources.
///
/// Once the target CPU comes around to service it it will find a cleared
/// vector and complain. While the spurious interrupt is harmless, the full
/// release of resources might prevent the interrupt from being acknowledged
/// which keeps the hardware in a weird state.
///
/// Verify that the corresponding Remote-IRR bits are clear.
#[unsafe(link_section = ".text.slate_distinct.io_apic.ioapic_irq_get_chip_state")]
extern "C" fn ioapic_irq_get_chip_state(irqd: *mut irq_data, which: u32, state: *mut bool) -> i32 {
let mcd: *mut mp_chip_data = (unsafe { (*irqd).chip_data }) as *mut mp_chip_data;
let mut rentry: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(rentry) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut p: *mut irq_pin_list = unsafe { core::mem::zeroed() };
if (which as u32) != 1 {
return 22i32.wrapping_neg();
}
unsafe {
*state = false;
}
let mut __UNIQUE_ID_guard_648: __SlateRecord_spinlock_h_558 = class_raw_spinlock_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_648) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_649: *mut __SlateRecord_spinlock_h_558 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_558;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_649) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
p = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*mcd).irq_2_pin) });
while list_entry_is_head!(p, unsafe { core::ptr::addr_of_mut!((*mcd).irq_2_pin) }, list) == 0 {
rentry = __ioapic_read_entry(unsafe { (*p).apic }, unsafe { (*p).pin });
// The remote IRR is only valid in level trigger mode. It's
// meaning is undefined for edge triggered interrupts and
// irrelevant because the IO-APIC treats them as fire and
// forget.
if ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(rentry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_irr() }) as i32) != 0 && ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(rentry.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_is_level() }) as i32) != 0 {
unsafe {
*state = true;
}
break;
}
p = list_next_entry__2!(p, list);
}
0
}

fn init_IO_APIC_traps() {
let mut irq: u32 = 0;
let mut __nr_irqs__: u32 = unsafe { irq_get_nr_irqs() };
while __nr_irqs__ != 0 {
irq = unsafe { irq_get_next_irq(0u32) };
while irq < __nr_irqs__ {
let cfg: *mut irq_cfg = unsafe { irq_cfg(irq) };
if IO_APIC_IRQ!(irq) && !cfg.is_null() && (unsafe { (*cfg).vector }) == 0 {
// Hmm.. We don't have an entry for this, so
// default to an old-fashioned 8259 interrupt if we
// can. Otherwise set the dummy interrupt chip.
if irq < (nr_legacy_irqs() as u32) {
unsafe { unsafe { (*legacy_pic).make_irq }.unwrap()(irq) };
} else {
unsafe { irq_set_chip(irq, core::ptr::addr_of_mut!(no_irq_chip) as *const irq_chip) };
}
}
irq = unsafe { irq_get_next_irq(irq.wrapping_add(1)) };
}
__nr_irqs__ = 0;
}
}

/// The local APIC irq-chip implementation:
#[unsafe(link_section = ".text.slate_distinct.io_apic.mask_lapic_irq")]
extern "C" fn mask_lapic_irq(data: *mut irq_data) {
let v: u64 = apic_read(848u32) as u64;
apic_write(848u32, (v | 65536u64) as u32);
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.unmask_lapic_irq")]
extern "C" fn unmask_lapic_irq(data: *mut irq_data) {
let v: u64 = apic_read(848u32) as u64;
apic_write(848u32, (v & 18446744073709486079u64) as u32);
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.ack_lapic_irq")]
extern "C" fn ack_lapic_irq(data: *mut irq_data) {
apic_eoi();
}

#[unsafe(link_section = ".data..read_mostly")]
static mut lapic_chip: irq_chip = {
let mut __t0: irq_chip = unsafe { core::mem::zeroed() };
__t0.name = (b"local-APIC\0".as_ptr() as *mut u8) as *const u8;
__t0.irq_ack = Some(ack_lapic_irq);
__t0.irq_mask = Some(mask_lapic_irq);
__t0.irq_unmask = Some(unmask_lapic_irq);
__t0
};

fn lapic_register_intr(irq: i32) {
irq_clear_status_flags(irq as u32, 256u64);
unsafe { irq_set_chip_and_handler_name(irq as u32, core::ptr::addr_of_mut!(lapic_chip) as *const irq_chip, unsafe { core::mem::transmute::<*const (), Option<unsafe extern "C" fn(*mut irq_desc)>>(handle_edge_irq as *const ()) }, c"edge".as_ptr() as *const u8) };
}

/// This looks a bit hackish but it's about the only one way of sending
/// a few INTA cycles to 8259As and any associated glue logic.  ICR does
/// not support the ExtINT mode, unfortunately.  We need to send these
/// cycles as some i82489DX-based boards have glue logic that keeps the
/// 8259A interrupt line asserted until INTA.  --macro
#[unsafe(link_section = ".init.text")]
fn unlock_ExtINT_logic() {
let mut save_control: u8 = 0;
let mut save_freq_select: u8 = 0;
let mut entry0: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry0) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut entry1: IO_APIC_route_entry = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(entry1) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
}
let mut apic__0: i32 = 0;
let pin: i32 = find_isa_irq_pin(8i32, 0i32);
let mut i: i32 = 100;
let mut apic_id: u32 = 0;
if pin == 1i32.wrapping_neg() {
let mut __v0: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("650: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 650b - ., 3; .popsection\n# {0}", const 650 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_unlock_ExtINT_logic_1, sym __slate_asm_string_unlock_ExtINT_logic_2, const 1952 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("651: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 651b - ., 4; .popsection\n# {0}", const 651 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
return;
}
apic__0 = find_isa_irq_apic(8i32, 0i32);
if apic__0 == 1i32.wrapping_neg() {
let mut __v5: i64;
let __ret_warn_on: i32 = true as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | (2i32 | BUGFLAG_TAINT!(9i32));
unsafe {
core::arch::asm!("652: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 652b - ., 3; .popsection\n# {0}", const 652 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_unlock_ExtINT_logic_6, sym __slate_asm_string_unlock_ExtINT_logic_7, const 1957 as i32, const 2307 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("653: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 653b - ., 4; .popsection\n# {0}", const 653 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
return;
}
entry0 = ioapic_read_entry(apic__0, pin);
clear_IO_APIC_pin(apic__0 as u32, pin as u32);
apic_id = read_apic_id();
unsafe { memset(core::ptr::addr_of_mut!(entry1) as *mut (), 0i32, core::mem::size_of::<IO_APIC_route_entry>() as u64) };
let __t0 = 1u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t1.__set_dest_mode_logical(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t1);
}
let __t2 = 0u64;
unsafe {
let mut __t3 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t3.__set_masked(__t2);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t3);
}
let __t4 = (apic_id & 255) as u64;
unsafe {
let mut __t5 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t5.__set_destid_0_7(__t4);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t5);
}
let __t6 = (apic_id >> 8i32) as u64;
unsafe {
let mut __t7 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t7.__set_virt_destid_8_14(__t6);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t7);
}
let __t8 = 7u64;
unsafe {
let mut __t9 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t9.__set_delivery_mode(__t8);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t9);
}
let __t10 = ((unsafe { core::ptr::read_unaligned(core::ptr::addr_of!(entry0.__slate_anon_0.__slate_anon_0.__slate_bits_0)).__get_active_low() }) as i32) as u64;
unsafe {
let mut __t11 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t11.__set_active_low(__t10);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t11);
}
let __t12 = 0u64;
unsafe {
let mut __t13 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t13.__set_is_level(__t12);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t13);
}
let __t14 = 0u64;
unsafe {
let mut __t15 = core::ptr::read_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t15.__set_vector(__t14);
core::ptr::write_unaligned(core::ptr::addr_of_mut!(entry1.__slate_anon_0.__slate_anon_0.__slate_bits_0), __t15);
}
ioapic_write_entry(apic__0, pin, entry1);
save_control = CMOS_READ!(11i32);
save_freq_select = CMOS_READ!(10i32);
CMOS_WRITE!((save_freq_select as i32) & !15 | 6, 10i32);
CMOS_WRITE!((save_control as i32) | 64, 11i32);
loop {
let __v15: i32 = i;
i = __v15.wrapping_sub(1);
if __v15 <= 0 {
break;
}
mdelay!(10i32);
if (CMOS_READ!(12i32) as i32) & 64 == 64 {
i = i.wrapping_sub(10);
}
}
CMOS_WRITE__2!(save_control, 11i32);
CMOS_WRITE__2!(save_freq_select, 10i32);
clear_IO_APIC_pin(apic__0 as u32, pin as u32);
ioapic_write_entry(apic__0, pin, entry0);
}

#[unsafe(link_section = ".init.data")]
static mut disable_timer_pin_1: i32 = unsafe { core::mem::zeroed() };

/// Actually the next is obsolete, but keep it for paranoid reasons -AK
#[unsafe(link_section = ".init.text")]
extern "C" fn disable_timer_pin_setup(arg: *mut u8) -> i32 {
unsafe {
disable_timer_pin_1 = 1;
}
0
}

#[unsafe(link_section = ".init.rodata")]
static __setup_str_disable_timer_pin_setup: [u8; 20] = [100, 105, 115, 97, 98, 108, 101, 95, 116, 105, 109, 101, 114, 95, 112, 105, 110, 95, 49, 0];

#[used(compiler)]
#[unsafe(link_section = ".init.setup")]
static mut __setup_disable_timer_pin_setup: obs_kernel_param = {
let mut __t0: obs_kernel_param = unsafe { core::mem::zeroed() };
__t0.str = unsafe { core::ptr::addr_of!(__setup_str_disable_timer_pin_setup) as *const u8 };
__t0.setup_func = Some(disable_timer_pin_setup);
__t0.early = 1i32;
__t0
};

#[unsafe(link_section = ".init.text")]
fn mp_alloc_timer_irq(ioapic: i32, pin: i32) -> i32 {
let domain: *mut irq_domain = mp_ioapic_irqdomain(ioapic);
let mut irq: i32 = 1i32.wrapping_neg();
if !domain.is_null() {
let mut info: irq_alloc_info = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(info) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 72usize);
}
ioapic_set_alloc_attr(core::ptr::addr_of_mut!(info), 1i32.wrapping_neg(), 0i32, 0i32);
info.devid = mpc_ioapic_id(ioapic) as u32;
unsafe {
info.__slate_anon_7.ioapic.pin = pin;
}
let mut __UNIQUE_ID_guard_654: __SlateRecord_mutex_h_253 = class_mutex_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_mutex) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_654) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_mutex_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_655: *mut __SlateRecord_mutex_h_253 = ((unsafe { core::ptr::addr_of_mut!(ioapic_mutex) }) as u64) as *mut __SlateRecord_mutex_h_253;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_655) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_mutex_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
irq = alloc_isa_irq_from_domain(domain, 0i32, ioapic, pin, core::ptr::addr_of_mut!(info));
}
irq
}

#[unsafe(link_section = ".init.text")]
fn replace_pin_at_irq_node(data: *mut mp_chip_data, node: i32, oldapic: i32, oldpin: i32, newapic: i32, newpin: i32) {
let mut entry: *mut irq_pin_list = unsafe { core::mem::zeroed() };
entry = list_first_entry__2!(unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) });
while list_entry_is_head!(entry, unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }, list) == 0 {
if (unsafe { (*entry).apic }) == oldapic && (unsafe { (*entry).pin }) == oldpin {
unsafe {
(*entry).apic = newapic;
}
unsafe {
(*entry).pin = newpin;
}
return;
}
entry = list_next_entry__2!(entry, list);
}
// Old apic/pin didn't exist, so just add a new one
add_pin_to_irq_node(data, node, newapic, newpin);
}

/// This code may look a bit paranoid, but it's supposed to cooperate with
/// a wide range of boards and BIOS bugs.  Fortunately only the timer IRQ
/// is so screwy.  Thanks to Brian Perkins for testing/hacking this beast
/// fanatically on his truly buggy board.
#[unsafe(link_section = ".init.text")]
fn check_timer() {
let mut __v12: bool = unsafe { core::mem::zeroed() };
// for edge trigger, it's already unmasked,
// so only need to unmask if it is level-trigger
// do we really have level trigger timer?
let mut idx: i32 = unsafe { core::mem::zeroed() };
let mut no_pin1: i32 = unsafe { core::mem::zeroed() };
let mut pin2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_16: *mut i32 = core::ptr::addr_of_mut!(pin2) as *mut i32;
let mut apic2: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_17: *mut i32 = core::ptr::addr_of_mut!(apic2) as *mut i32;
let mut pin1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_18: *mut i32 = core::ptr::addr_of_mut!(pin1) as *mut i32;
let mut apic1: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_19: *mut i32 = core::ptr::addr_of_mut!(apic1) as *mut i32;
let mut node: i32 = unsafe { core::mem::zeroed() };
let mut data: *mut mp_chip_data = unsafe { core::mem::zeroed() };
let mut irq_data: *mut irq_data = unsafe { core::mem::zeroed() };
unsafe {
irq_data = irq_get_irq_data(0u32);
data = (*irq_data).chip_data as *mut mp_chip_data;
let cfg: *mut irq_cfg = irqd_cfg(irq_data);
node = cpu_to_node(0i32);
let __slate_auto_bytes = __slate_slot_19 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_18 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_2 = __slate_slot_17 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 4usize);
let __slate_auto_bytes = __slate_slot_16 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
no_pin1 = 0;
if unsafe { global_clock_event }.is_null() {
return;
} else {
local_irq_disable!();
// get/set the timer IRQ vector:
unsafe { (*legacy_pic).mask }.unwrap()(0u32);
// As IRQ0 is to be enabled in the 8259A, the virtual
// wire has to be disabled in the local APIC.  Also
// timer interrupts need to be acknowledged manually in
// the 8259A for the i82489DX when using the NMI
// watchdog as that APIC treats NMIs as level-triggered.
// The AEOI mode will finish them in the 8259A
// automatically.
apic_write(848u32, 67328u32);
unsafe { (*legacy_pic).init }.unwrap()(1i32);
*__slate_slot_18 = find_isa_irq_pin(0i32, 0i32);
*__slate_slot_19 = find_isa_irq_apic(0i32, 0i32);
*__slate_slot_16 = ioapic_i8259.pin;
*__slate_slot_17 = ioapic_i8259.apic;
while false {
}
_printk(c"\x016..TIMER: vector=0x%02X apic1=%d pin1=%d apic2=%d pin2=%d\n".as_ptr() as *const u8, (*cfg).vector, *__slate_slot_19, *__slate_slot_18, *__slate_slot_17, *__slate_slot_16);
// Some BIOS writers are clueless and report the ExtINTA
// I/O APIC input from the cascaded 8259A as the timer
// interrupt input.  So just in case, if only one pin
// was found above, try it both directly and through the
// 8259A.
if *__slate_slot_18 == 1i32.wrapping_neg() {
panic_if_irq_remap(c"[Firmware Bug]: Timer not connected to IO-APIC".as_ptr() as *const u8);
*__slate_slot_18 = *__slate_slot_16;
*__slate_slot_19 = *__slate_slot_17;
no_pin1 = 1;
} else {
if *__slate_slot_16 == 1i32.wrapping_neg() {
*__slate_slot_16 = *__slate_slot_18;
*__slate_slot_17 = *__slate_slot_19;
}
}
'__join_0: {
if *__slate_slot_18 != 1i32.wrapping_neg() {
// Ok, does IRQ0 through the IOAPIC work?
if no_pin1 != 0 {
mp_alloc_timer_irq(*__slate_slot_19, *__slate_slot_18);
} else {
idx = find_irq_entry(*__slate_slot_19, *__slate_slot_18, 0i32);
if idx != 1i32.wrapping_neg() {
__v12 = irq_is_level(idx);
} else {
__v12 = false;
}
if __v12 {
unmask_ioapic_irq(irq_get_irq_data(0u32));
}
}
irq_domain_deactivate_irq(irq_data);
irq_domain_activate_irq(irq_data, false);
if timer_irq_works() != 0 {
if disable_timer_pin_1 > 0 {
clear_IO_APIC_pin(0u32, *__slate_slot_18 as u32);
break '__join_0;
} else {
break '__join_0;
}
} else {
panic_if_irq_remap(c"timer doesn't work through Interrupt-remapped IO-APIC".as_ptr() as *const u8);
clear_IO_APIC_pin(*__slate_slot_19 as u32, *__slate_slot_18 as u32);
if no_pin1 == 0 {
while false {
}
_printk(c"\x013..MP-BIOS bug: 8254 timer not connected to IO-APIC\n".as_ptr() as *const u8);
}
while false {
}
_printk(c"\x016...trying to set up timer (IRQ0) through the 8259A ...\n".as_ptr() as *const u8);
while false {
}
_printk(c"\x016..... (found apic %d pin %d) ...\n".as_ptr() as *const u8, *__slate_slot_17, *__slate_slot_16);
// legacy devices should be connected to IO APIC #0
replace_pin_at_irq_node(data, node, *__slate_slot_19, *__slate_slot_18, *__slate_slot_17, *__slate_slot_16);
irq_domain_deactivate_irq(irq_data);
irq_domain_activate_irq(irq_data, false);
unsafe { (*legacy_pic).unmask }.unwrap()(0u32);
if timer_irq_works() != 0 {
while false {
}
_printk(c"\x016....... works.\n".as_ptr() as *const u8);
break '__join_0;
} else {
// Cleanup, just in case ...
unsafe { (*legacy_pic).mask }.unwrap()(0u32);
clear_IO_APIC_pin(*__slate_slot_17 as u32, *__slate_slot_16 as u32);
while false {
}
_printk(c"\x016....... failed.\n".as_ptr() as *const u8);
}
}
}
while false {
}
_printk(c"\x016...trying to set up timer as Virtual Wire IRQ...\n".as_ptr() as *const u8);
lapic_register_intr(0i32);
apic_write(848u32, 0 | (*cfg).vector); // Fixed mode
unsafe { (*legacy_pic).unmask }.unwrap()(0u32);
if timer_irq_works() != 0 {
while false {
}
_printk(c"\x016..... works.\n".as_ptr() as *const u8);
} else {
unsafe { (*legacy_pic).mask }.unwrap()(0u32);
apic_write(848u32, 65536u32 | (*cfg).vector);
while false {
}
_printk(c"\x016..... failed.\n".as_ptr() as *const u8);
while false {
}
_printk(c"\x016...trying to set up timer as ExtINT IRQ...\n".as_ptr() as *const u8);
unsafe { (*legacy_pic).init }.unwrap()(0i32);
unsafe { (*legacy_pic).make_irq }.unwrap()(0u32);
apic_write(848u32, 1792u32);
unsafe { (*legacy_pic).unmask }.unwrap()(0u32);
unlock_ExtINT_logic();
if timer_irq_works() != 0 {
while false {
}
_printk(c"\x016..... works.\n".as_ptr() as *const u8);
} else {
while false {
}
_printk(c"\x016..... failed :\n".as_ptr() as *const u8);
if apic_is_x2apic_enabled() {
while false {
}
_printk(c"\x016Perhaps problem with the pre-enabled x2apic mode\nTry booting with x2apic and interrupt-remapping disabled in the bios.\n".as_ptr() as *const u8);
}
panic(c"IO-APIC + timer doesn't work!  Boot with apic=debug and send a report.  Then try booting with the 'noapic' option.\n".as_ptr() as *const u8);
core::hint::unreachable_unchecked();
}
}
}
local_irq_enable!();
}
}
}

// Traditionally ISA IRQ2 is the cascade IRQ, and is not available
// to devices.  However there may be an I/O APIC pin available for
// this interrupt regardless.  The pin may be left unconnected, but
// typically it will be reused as an ExtINT cascade interrupt for
// the master 8259A.  In the MPS case such a pin will normally be
// reported as an ExtINT interrupt in the MP table.  With ACPI
// there is no provision for ExtINT interrupts, and in the absence
// of an override it would be treated as an ordinary ISA I/O APIC
// interrupt, that is edge-triggered and unmasked by default.  We
// used to do this, but it caused problems on some systems because
// of the NMI watchdog and sometimes IRQ0 of the 8254 timer using
// the same ExtINT cascade interrupt to drive the local APIC of the
// bootstrap processor.  Therefore we refrain from routing IRQ2 to
// the I/O APIC in all cases now.  No actual device should request
// it anyway.  --macro
fn mp_irqdomain_create(ioapic: i32) -> i32 {
let gsi_cfg: *mut mp_ioapic_gsi = mp_ioapic_gsi_routing(ioapic);
let hwirqs: i32 = mp_ioapic_pin_count(ioapic);
let ip: *mut ioapic = unsafe { core::ptr::addr_of_mut!(ioapics.0) as *mut ioapic }.wrapping_offset(ioapic as isize);
let cfg: *mut ioapic_domain_cfg = unsafe { core::ptr::addr_of_mut!((*ip).irqdomain_cfg) };
let mut r#fn: *mut fwnode_handle = unsafe { core::mem::zeroed() };
let mut fwspec: irq_fwspec = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(fwspec) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 80usize);
}
if ((unsafe { (*cfg).r#type }) as u32) == 0 {
return 0;
}
// Handle device tree enumerated APICs proper
if !unsafe { (*cfg).dev }.is_null() {
r#fn = core::ptr::null_mut::<fwnode_handle>();
} else {
r#fn = irq_domain_alloc_named_id_fwnode(c"IO-APIC".as_ptr() as *const u8, mpc_ioapic_id(ioapic));
if r#fn.is_null() {
return 12i32.wrapping_neg();
}
}
fwspec.fwnode = r#fn;
fwspec.param_count = 1;
fwspec.param[0] = mpc_ioapic_id(ioapic) as u32;
let parent: *mut irq_domain = unsafe { irq_find_matching_fwspec(core::ptr::addr_of_mut!(fwspec), 2u32) };
if parent.is_null() {
if unsafe { (*cfg).dev }.is_null() {
unsafe { irq_domain_free_fwnode(r#fn) };
}
return 19i32.wrapping_neg();
}
unsafe {
(*ip).irqdomain = irq_domain_create_hierarchy(parent, 0u32, hwirqs as u32, r#fn, (*cfg).ops, (ioapic as i64) as *mut ());
}
if unsafe { (*ip).irqdomain }.is_null() {
// Release fw handle if it was allocated above
if unsafe { (*cfg).dev }.is_null() {
unsafe { irq_domain_free_fwnode(r#fn) };
}
return 12i32.wrapping_neg();
}
if ((unsafe { (*cfg).r#type }) as u32) == 1 || ((unsafe { (*cfg).r#type }) as u32) == 2 {
let mut __v0: u32;
let __UNIQUE_ID_x__656: u32 = unsafe { ioapic_dynirq_base };
let __UNIQUE_ID_y__657: u32 = unsafe { (*gsi_cfg).gsi_end }.wrapping_add(1);
loop {
if __types_ok__4!(__UNIQUE_ID_x__656, __UNIQUE_ID_y__657) == 0 {
unsafe { __compiletime_assert_658() };
unsafe { core::hint::unreachable_unchecked() };
}
if true {
break;
}
}
__v0 = if __UNIQUE_ID_x__656 > __UNIQUE_ID_y__657 { __UNIQUE_ID_x__656 } else { __UNIQUE_ID_y__657 };
unsafe {
ioapic_dynirq_base = __v0;
}
}
0
}

fn ioapic_destroy_irqdomain(idx: i32) {
let cfg: *mut ioapic_domain_cfg = unsafe { core::ptr::addr_of_mut!(ioapics.0[idx as usize].irqdomain_cfg) };
let r#fn: *mut fwnode_handle = unsafe { (*ioapics.0[idx as usize].irqdomain).fwnode };
if !unsafe { ioapics.0[idx as usize].irqdomain }.is_null() {
unsafe { irq_domain_remove(ioapics.0[idx as usize].irqdomain) };
if unsafe { (*cfg).dev }.is_null() {
unsafe { irq_domain_free_fwnode(r#fn) };
}
unsafe {
ioapics.0[idx as usize].irqdomain = core::ptr::null_mut::<irq_domain>();
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn setup_IO_APIC() {
let mut ioapic: i32 = 0;
if (unsafe { ioapic_is_disabled }) || (unsafe { nr_ioapics }) == 0 {
return;
}
unsafe {
io_apic_irqs = if nr_legacy_irqs() != 0 { 18446744073709551611u64 } else { !0 };
}
loop {
if 1 <= unsafe { apic_verbosity } {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016ENABLING IO-APIC IRQs\n".as_ptr() as *const u8) };
}
if true {
break;
}
}
while ioapic < unsafe { nr_ioapics } {
loop {
if unlikely!(mp_irqdomain_create(ioapic)) != 0 {
loop {
unsafe {
core::arch::asm!("659: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 659b - ., 3; .popsection\n# {0}", const 659 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_setup_IO_APIC_3, sym __slate_asm_string_setup_IO_APIC_4, const 2283 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
ioapic = ioapic.wrapping_add(1);
}
// Set up IO-APIC IRQ routing.
unsafe { unsafe { x86_init.mpparse.setup_ioapic_ids }.unwrap()() };
unsafe { sync_Arb_IDs() };
setup_IO_APIC_irqs();
init_IO_APIC_traps();
if nr_legacy_irqs() != 0 {
check_timer();
}
unsafe {
ioapic_initialized = 1;
}
}

fn resume_ioapic_id(ioapic_idx: i32) {
let mut reg_00: IO_APIC_reg_00 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(reg_00) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut __UNIQUE_ID_guard_660: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_660) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_661: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_661) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe {
reg_00.raw = io_apic_read(ioapic_idx as u32, 0u32);
}
if ((unsafe { reg_00.bits.__slate_bits_0.__get_ID() }) as i32) != mpc_ioapic_id(ioapic_idx) {
unsafe {
let __t0 = mpc_ioapic_id(ioapic_idx) as u32;
reg_00.bits.__slate_bits_0.__set_ID(__t0);
}
io_apic_write(ioapic_idx as u32, 0u32, unsafe { reg_00.raw });
}
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.ioapic_suspend")]
extern "C" fn ioapic_suspend(data: *mut ()) -> i32 {
save_ioapic_entries()
}

#[unsafe(link_section = ".text.slate_distinct.io_apic.ioapic_resume")]
extern "C" fn ioapic_resume(data: *mut ()) {
let mut ioapic_idx: i32 = unsafe { nr_ioapics }.wrapping_sub(1);
while ioapic_idx >= 0 {
resume_ioapic_id(ioapic_idx);
ioapic_idx = ioapic_idx.wrapping_sub(1);
}
restore_ioapic_entries();
}

static ioapic_syscore_ops: __SlateConst<syscore_ops> = __SlateConst({
let mut __t0: syscore_ops = unsafe { core::mem::zeroed() };
__t0.suspend = Some(ioapic_suspend);
__t0.resume = Some(ioapic_resume);
__t0
});

static mut ioapic_syscore: syscore = {
let mut __t0: syscore = unsafe { core::mem::zeroed() };
__t0.ops = unsafe { core::ptr::addr_of!(ioapic_syscore_ops.0) };
__t0
};

#[unsafe(link_section = ".init.text")]
extern "C" fn ioapic_init_ops() -> i32 {
unsafe { register_syscore(core::ptr::addr_of_mut!(ioapic_syscore)) };
0
}

#[used(compiler)]
#[unsafe(link_section = ".discard.addressable")]
static mut __UNIQUE_ID_addressable_ioapic_init_ops_663: *mut () = unsafe { core::mem::transmute::<Option<unsafe extern "C" fn() -> i32>, *mut ()>(Some(ioapic_init_ops)) };

fn io_apic_get_redir_entries(ioapic: i32) -> i32 {
let mut reg_01: IO_APIC_reg_01 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(reg_01) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut __UNIQUE_ID_guard_664: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_664) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_665: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_665) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe {
reg_01.raw = io_apic_read(ioapic as u32, 1u32);
}
// The register returns the maximum index redir index supported,
// which is one less than the total number of redir entries.
((unsafe { reg_01.bits.__slate_bits_0.__get_entries() }) as i32).wrapping_add(1)
}

#[unsafe(no_mangle)]
extern "C" fn arch_dynirq_lower_bound(from: u32) -> u32 {
let mut ret: u32 = 0;
// dmar_alloc_hwirq() may be called before setup_IO_APIC(), so use
// gsi_top if ioapic_dynirq_base hasn't been initialized yet.
let __v0: u32 = unsafe { ioapic_dynirq_base };
ret = if __v0 != 0 { __v0 } else { unsafe { gsi_top } };
// For DT enabled machines ioapic_dynirq_base is irrelevant and
// always 0. gsi_top can be 0 if there is no IO/APIC registered.
// 0 is an invalid interrupt number for dynamic allocations. Return
// @from instead.
let __v1: u32 = ret;
if __v1 != 0 { __v1 } else { from }
}

fn io_apic_unique_id(idx: i32, id: u8) -> u8 {
let mut __UNIQUE_ID_unlock_669: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_0: *mut *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_669) as *mut *mut __SlateRecord_spinlock_h_594;
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_594;
let mut __UNIQUE_ID_unlock_667: core::mem::MaybeUninit<*mut __SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_3: *mut *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_667) as *mut *mut __SlateRecord_spinlock_h_594;
// Read the current id from the ioapic and keep it if
// available.
let mut scope: core::mem::MaybeUninit<__SlateRecord_spinlock_h_594> = core::mem::MaybeUninit::uninit();
let __slate_slot_4: *mut __SlateRecord_spinlock_h_594 = core::ptr::addr_of_mut!(scope) as *mut __SlateRecord_spinlock_h_594;
let mut i: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut i32 = core::ptr::addr_of_mut!(i) as *mut i32;
let mut new_id: core::mem::MaybeUninit<u8> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut u8 = core::ptr::addr_of_mut!(new_id) as *mut u8;
let mut used: core::mem::MaybeUninit<__SlateAlign16<[u64; 4]>> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut [u64; 4] = core::ptr::addr_of_mut!(used) as *mut [u64; 4];
let mut reg_00: core::mem::MaybeUninit<IO_APIC_reg_00> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut IO_APIC_reg_00 = core::ptr::addr_of_mut!(reg_00) as *mut IO_APIC_reg_00;
let __slate_cleanup_ptr_4 = __slate_slot_4;
let __slate_cleanup_armed_4 = core::cell::Cell::new(false);
let __slate_cleanup_4 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_4.get() {
unsafe { class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_4 as *mut _) };
}
});
let __slate_cleanup_ptr_3 = __slate_slot_3;
let __slate_cleanup_armed_3 = core::cell::Cell::new(false);
let __slate_cleanup_3 = ::slate_support::Guard(&mut || {
if __slate_cleanup_armed_3.get() {
unsafe { __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_3 as *mut _) };
}
});
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
let __slate_auto_bytes = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_1 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 32usize);
let __slate_auto_bytes_2 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 1usize);
let __slate_auto_bytes = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
bitmap_zero((*__slate_slot_9).as_mut_ptr() as *mut u64, 256u32);
*__slate_slot_7 = 0;
while *__slate_slot_7 < nr_ioapics {
__set_bit__4!(mpc_ioapic_id(*__slate_slot_7), (*__slate_slot_9).as_mut_ptr() as *mut u64);
let __v6: i32 = *__slate_slot_7;
*__slate_slot_7 = __v6.wrapping_add(1);
}
// Hand out the requested id if available
if test_bit__5!(id, (*__slate_slot_9).as_mut_ptr() as *mut u64) == 0 {
return id;
} else {
core::ptr::write(__slate_slot_4, class_raw_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(ioapic_lock)));
__slate_cleanup_armed_4.set(true);
core::ptr::write(__slate_slot_3, (core::ptr::addr_of_mut!(ioapic_lock) as u64) as *mut __SlateRecord_spinlock_h_594);
__slate_cleanup_armed_3.set(true);
if !class_raw_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_4))).is_null() || !class_raw_spinlock_irqsave_is_conditional {
if true {
(*__slate_slot_10).raw = io_apic_read(idx as u32, 0u32);
}
}
__slate_cleanup_armed_3.set(false);
__class_raw_spinlock_irqsave_cleanup_ctx(__slate_slot_3 as *mut _);
__slate_cleanup_armed_4.set(false);
class_raw_spinlock_irqsave_destructor(__slate_slot_4 as *mut _);
*__slate_slot_8 = ((*__slate_slot_10).bits.__slate_bits_0.__get_ID() as i32) as u8;
if test_bit__5!(*__slate_slot_8, (*__slate_slot_9).as_mut_ptr() as *mut u64) == 0 {
loop {
if 1 <= apic_verbosity {
while false {
}
_printk(c"\x016IOAPIC[%d]: Using reg apic_id %d instead of %d\n".as_ptr() as *const u8, idx, *__slate_slot_8 as i32, id as i32);
}
if true {
break;
}
}
return *__slate_slot_8;
} else {
// Get the next free id and write it to the ioapic.
*__slate_slot_8 = find_first_zero_bit((*__slate_slot_9).as_mut_ptr() as *const u64, 256u64) as u8;
(*__slate_slot_10).bits.__slate_bits_0.__set_ID(*__slate_slot_8 as u32);
core::ptr::write(__slate_slot_1, class_raw_spinlock_irqsave_constructor(core::ptr::addr_of_mut!(ioapic_lock)));
__slate_cleanup_armed_1.set(true);
core::ptr::write(__slate_slot_0, (core::ptr::addr_of_mut!(ioapic_lock) as u64) as *mut __SlateRecord_spinlock_h_594);
__slate_cleanup_armed_0.set(true);
if !class_raw_spinlock_irqsave_lock_ptr(&mut (*core::ptr::addr_of_mut!(*__slate_slot_1))).is_null() || !class_raw_spinlock_irqsave_is_conditional {
if true {
io_apic_write(idx as u32, 0u32, (*__slate_slot_10).raw);
(*__slate_slot_10).raw = io_apic_read(idx as u32, 0u32);
}
}
__slate_cleanup_armed_0.set(false);
__class_raw_spinlock_irqsave_cleanup_ctx(__slate_slot_0 as *mut _);
__slate_cleanup_armed_1.set(false);
class_raw_spinlock_irqsave_destructor(__slate_slot_1 as *mut _);
// Sanity check
'__join_3: {
loop {
if unlikely__2!(((*__slate_slot_10).bits.__slate_bits_0.__get_ID() as i32) != (*__slate_slot_8 as i32)) != 0 {
break '__join_3;
} else {
if true {
break;
}
}
}
return *__slate_slot_8;
}
core::arch::asm!("670: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 670b - ., 3; .popsection\n# {0}", const 670 as i32, options(nostack, att_syntax));
loop {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_io_apic_unique_id_11, sym __slate_asm_string_io_apic_unique_id_12, const 2476 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
if true {
core::hint::unreachable_unchecked();
}
}
}
}
}
unsafe { core::mem::zeroed() }
}

fn io_apic_get_version(ioapic: i32) -> i32 {
let mut reg_01: IO_APIC_reg_01 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(reg_01) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut __UNIQUE_ID_guard_671: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_671) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_672: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_672) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
unsafe {
reg_01.raw = io_apic_read(ioapic as u32, 1u32);
}
(unsafe { reg_01.bits.__slate_bits_0.__get_version() }) as i32
}

// This function updates target affinity of IOAPIC interrupts to include
// the CPUs which came online during SMP bringup.
static mut ioapic_resources: *mut resource = unsafe { core::mem::zeroed() };

#[unsafe(link_section = ".init.text")]
fn ioapic_setup_resources() -> *mut resource {
let mut n: u64 = 0;
let mut mem: *mut u8 = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
if (unsafe { nr_ioapics }) == 0 {
return core::ptr::null_mut::<resource>();
}
n = 11u64.wrapping_add(core::mem::size_of::<resource>() as u64);
n = n.wrapping_mul((unsafe { nr_ioapics }) as u64);
mem = memblock_alloc_or_panic!(n, 64i32) as *mut u8;
let res: *mut resource = mem as *mut resource;
let __v2: *mut u8 = mem;
mem = __v2.wrapping_offset((core::mem::size_of::<resource>() as u64).wrapping_mul((unsafe { nr_ioapics }) as u64) as isize);
while i < unsafe { nr_ioapics } {
unsafe {
(*res.wrapping_offset(i as isize)).name = mem as *const u8;
}
unsafe {
(*res.wrapping_offset(i as isize)).flags = 2147484160u64;
}
unsafe { snprintf(mem, 11u64, c"IOAPIC %u".as_ptr() as *const u8, i) };
let __v4: *mut u8 = mem;
mem = __v4.wrapping_offset(11 as isize);
unsafe {
ioapics.0[i as usize].iomem_res = res.wrapping_offset(i as isize);
}
i = i.wrapping_add(1);
}
unsafe {
ioapic_resources = res;
}
res
}

fn io_apic_set_fixmap(idx: u32, phys: u64) {
let mut __slate_compound_0: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_1: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_2: pgprot = unsafe { core::mem::zeroed() };
let mut flags: pgprot = unsafe { *{
__slate_compound_0 = pgprot { pgprot: (_AT!(u64, 1i32) << 0i32 | _AT!(u64, 1i32) << 1i32 | 0 | _AT!(u64, 1i32) << 5i32 | _AT!(u64, 1i32) << 63i32 | _AT!(u64, 1i32) << 6i32 | 0 | _AT!(u64, 1i32) << 8i32 | cachemode2protval(3u32)) & __default_kernel_pte_mask };
core::ptr::addr_of_mut!(__slate_compound_0)
} };
// Ensure fixmaps for IO-APIC MMIO respect memory encryption pgprot
// bits, just like normal ioremap():
if cc_platform_has(2u32) {
if unsafe { unsafe { x86_platform.hyper.is_private_mmio }.unwrap()(phys) } {
flags = unsafe { *{
__slate_compound_1 = pgprot { pgprot: cc_mkenc(pgprot_val!(flags)) };
core::ptr::addr_of_mut!(__slate_compound_1)
} };
} else {
flags = unsafe { *{
__slate_compound_2 = pgprot { pgprot: cc_mkdec(pgprot_val!(flags)) };
core::ptr::addr_of_mut!(__slate_compound_2)
} };
}
}
__set_fixmap(idx, phys, flags);
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn io_apic_init_mappings() {
let mut ioapic_phys: u64 = 0;
let mut idx: u64 = 516;
let mut ioapic_res: *mut resource = unsafe { core::mem::zeroed() };
let mut i: i32 = 0;
ioapic_res = ioapic_setup_resources();
while i < unsafe { nr_ioapics } {
if (unsafe { smp_found_config }) != 0 {
ioapic_phys = mpc_ioapic_addr(i) as u64;
} else {
ioapic_phys = memblock_alloc_or_panic__3!(_AC!() << 12i32, _AC!() << 12i32) as u64;
ioapic_phys = __pa!(ioapic_phys);
}
io_apic_set_fixmap(idx as u32, ioapic_phys);
loop {
if 1 <= unsafe { apic_verbosity } {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x016mapped IOAPIC to %08lx (%08lx)\n".as_ptr() as *const u8, __fix_to_virt__3!(idx).wrapping_add(ioapic_phys & !(!(_AC!() << 12i32).wrapping_sub(1))), ioapic_phys) };
}
if true {
break;
}
}
idx = idx.wrapping_add(1);
unsafe {
(*ioapic_res).start = ioapic_phys;
}
unsafe {
(*ioapic_res).end = ioapic_phys.wrapping_add(1024).wrapping_sub(1);
}
let __v5: *mut resource = ioapic_res;
ioapic_res = __v5.wrapping_offset(1 as isize);
i = i.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".init.text")]
extern "C" fn ioapic_insert_resources() {
let mut r: *mut resource = unsafe { ioapic_resources };
let mut i: i32 = 0;
if r.is_null() {
if (unsafe { nr_ioapics }) > 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x013IO APIC resources couldn't be allocated.\n".as_ptr() as *const u8) };
}
return;
}
while i < unsafe { nr_ioapics } {
unsafe { insert_resource(core::ptr::addr_of_mut!(iomem_resource), r) };
let __v2: *mut resource = r;
r = __v2.wrapping_offset(1 as isize);
i = i.wrapping_add(1);
}
}

#[unsafe(no_mangle)]
extern "C" fn mp_find_ioapic(gsi: u32) -> i32 {
// Find the IOAPIC that manages this GSI.
let mut i: i32 = 0;
if (unsafe { nr_ioapics }) == 0 {
return 1i32.wrapping_neg();
}
while i < unsafe { nr_ioapics } {
let gsi_cfg: *mut mp_ioapic_gsi = mp_ioapic_gsi_routing(i);
if gsi >= unsafe { (*gsi_cfg).gsi_base } && gsi <= unsafe { (*gsi_cfg).gsi_end } {
return i;
}
i = i.wrapping_add(1);
}
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x013ERROR: Unable to locate IOAPIC for GSI %d\n".as_ptr() as *const u8, gsi) };
1i32.wrapping_neg()
}

#[unsafe(no_mangle)]
extern "C" fn mp_find_ioapic_pin(ioapic: i32, gsi: u32) -> i32 {
let mut __v0: i64;
{
let __ret_warn_on: i32 = (ioapic < 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("673: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 673b - ., 3; .popsection\n# {0}", const 673 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mp_find_ioapic_pin_1, sym __slate_asm_string_mp_find_ioapic_pin_2, const 2627 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("674: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 674b - ., 4; .popsection\n# {0}", const 674 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v0 = (__ret_warn_on != 0) as i64;
}
if __v0 != 0 {
return 1i32.wrapping_neg();
}
let gsi_cfg: *mut mp_ioapic_gsi = mp_ioapic_gsi_routing(ioapic);
let mut __v5: i64;
let __ret_warn_on: i32 = (gsi > unsafe { (*gsi_cfg).gsi_end }) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("675: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 675b - ., 3; .popsection\n# {0}", const 675 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mp_find_ioapic_pin_6, sym __slate_asm_string_mp_find_ioapic_pin_7, const 2631 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("676: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 676b - ., 4; .popsection\n# {0}", const 676 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
if __v5 != 0 {
return 1i32.wrapping_neg();
}
gsi.wrapping_sub(unsafe { (*gsi_cfg).gsi_base }) as i32
}

fn bad_ioapic_register(idx: i32) -> i32 {
let mut reg_00: IO_APIC_reg_00 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(reg_00) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut reg_01: IO_APIC_reg_01 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(reg_01) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
let mut reg_02: IO_APIC_reg_02 = unsafe { core::mem::zeroed() };
unsafe {
let __slate_auto_bytes = (core::ptr::addr_of_mut!(reg_02) as *mut _) as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 4usize);
}
unsafe {
reg_00.raw = io_apic_read(idx as u32, 0u32);
}
unsafe {
reg_01.raw = io_apic_read(idx as u32, 1u32);
}
unsafe {
reg_02.raw = io_apic_read(idx as u32, 2u32);
}
if (unsafe { reg_00.raw }) == (1i32.wrapping_neg() as u32) && (unsafe { reg_01.raw }) == (1i32.wrapping_neg() as u32) && (unsafe { reg_02.raw }) == (1i32.wrapping_neg() as u32) {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014I/O APIC 0x%x registers return all ones, skipping!\n".as_ptr() as *const u8, mpc_ioapic_addr(idx)) };
return 1;
}
0
}

fn find_free_ioapic_entry() -> i32 {
for idx in 0i32..128i32 {
if (unsafe { ioapics.0[idx as usize].nr_registers }) == 0 {
return idx;
}
}
128
}

/// mp_register_ioapic - Register an IOAPIC device
/// @id:		hardware IOAPIC ID
/// @address:	physical address of IOAPIC register area
/// @gsi_base:	base of GSI associated with the IOAPIC
/// @cfg:	configuration information for the IOAPIC
#[unsafe(no_mangle)]
extern "C" fn mp_register_ioapic(id: i32, address: u32, gsi_base: u32, cfg: *mut ioapic_domain_cfg) -> i32 {
let mut __slate_compound_9: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_12: pgprot = unsafe { core::mem::zeroed() };
let mut __slate_compound_16: pgprot = unsafe { core::mem::zeroed() };
let hotplug: bool = (unsafe { ioapic_initialized }) != 0;
let mut gsi_cfg: *mut mp_ioapic_gsi = unsafe { core::mem::zeroed() };
let mut idx: i32 = 0;
let mut ioapic: i32 = 0;
let mut entries: i32 = 0;
let mut gsi_end: u32 = 0;
if address == 0 {
let mut __v0: i32;
while false {
}
__v0 = unsafe { _printk(c"\x014Bogus (zero) I/O APIC address found, skipping!\n".as_ptr() as *const u8) };
return 22i32.wrapping_neg();
}
while ioapic < unsafe { nr_ioapics } {
if (unsafe { ioapics.0[ioapic as usize].mp_config.apicaddr }) == address {
let mut __v2: i32;
while false {
}
__v2 = unsafe { _printk(c"\x014address 0x%x conflicts with IOAPIC%d\n".as_ptr() as *const u8, address, ioapic) };
return 17i32.wrapping_neg();
}
ioapic = ioapic.wrapping_add(1);
}
idx = find_free_ioapic_entry();
if idx >= 128 {
let mut __v7: i32;
while false {
}
__v7 = unsafe { _printk(c"\x014Max # of I/O APICs (%d) exceeded (found %d), skipping\n".as_ptr() as *const u8, 128i32, idx) };
return 28i32.wrapping_neg();
}
unsafe {
ioapics.0[idx as usize].mp_config.r#type = 2;
}
unsafe {
ioapics.0[idx as usize].mp_config.flags = 1;
}
unsafe {
ioapics.0[idx as usize].mp_config.apicaddr = address;
}
io_apic_set_fixmap(516i32.wrapping_add(idx) as u32, address as u64);
if bad_ioapic_register(idx) != 0 {
__set_fixmap(516i32.wrapping_add(idx) as u32, 0u64, unsafe { *{
__slate_compound_9 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_9)
} });
return 19i32.wrapping_neg();
}
unsafe {
ioapics.0[idx as usize].mp_config.apicid = io_apic_unique_id(idx, id as u8);
}
unsafe {
ioapics.0[idx as usize].mp_config.apicver = io_apic_get_version(idx) as u8;
}
// Build basic GSI lookup table to facilitate gsi->io_apic lookups
// and to prevent reprogramming of IOAPIC pins (PCI GSIs).
entries = io_apic_get_redir_entries(idx);
gsi_end = gsi_base.wrapping_add(entries as u32).wrapping_sub(1);
ioapic = 0;
'__slate_break_13: while ioapic < unsafe { nr_ioapics } {
gsi_cfg = mp_ioapic_gsi_routing(ioapic);
if gsi_base >= unsafe { (*gsi_cfg).gsi_base } && gsi_base <= unsafe { (*gsi_cfg).gsi_end } || gsi_end >= unsafe { (*gsi_cfg).gsi_base } && gsi_end <= unsafe { (*gsi_cfg).gsi_end } {
let mut __v10: i32;
while false {
}
__v10 = unsafe { _printk(c"\x014GSI range [%u-%u] for new IOAPIC conflicts with GSI[%u-%u]\n".as_ptr() as *const u8, gsi_base, gsi_end, (*gsi_cfg).gsi_base, (*gsi_cfg).gsi_end) };
__set_fixmap(516i32.wrapping_add(idx) as u32, 0u64, unsafe { *{
__slate_compound_12 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_12)
} });
return 28i32.wrapping_neg();
}
ioapic = ioapic.wrapping_add(1);
}
gsi_cfg = mp_ioapic_gsi_routing(idx);
unsafe {
(*gsi_cfg).gsi_base = gsi_base;
}
unsafe {
(*gsi_cfg).gsi_end = gsi_end;
}
unsafe {
ioapics.0[idx as usize].irqdomain = core::ptr::null_mut::<irq_domain>();
}
unsafe {
ioapics.0[idx as usize].irqdomain_cfg = *cfg;
}
// If mp_register_ioapic() is called during early boot stage when
// walking ACPI/DT tables, it's too early to create irqdomain,
// we are still using bootmem allocator. So delay it to setup_IO_APIC().
if hotplug {
if mp_irqdomain_create(idx) != 0 {
__set_fixmap(516i32.wrapping_add(idx) as u32, 0u64, unsafe { *{
__slate_compound_16 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_16)
} });
return 12i32.wrapping_neg();
}
alloc_ioapic_saved_registers(idx);
}
if (unsafe { (*gsi_cfg).gsi_end }) >= unsafe { gsi_top } {
unsafe {
gsi_top = unsafe { (*gsi_cfg).gsi_end }.wrapping_add(1);
}
}
if (unsafe { nr_ioapics }) <= idx {
unsafe {
nr_ioapics = idx.wrapping_add(1);
}
}
// Set nr_registers to mark entry present
unsafe {
ioapics.0[idx as usize].nr_registers = entries;
}
let mut __v17: i32;
while false {
}
__v17 = unsafe { _printk(c"\x016IOAPIC[%d]: apic_id %d, version %d, address 0x%x, GSI %d-%d\n".as_ptr() as *const u8, idx, mpc_ioapic_id(idx), mpc_ioapic_ver!(idx) as i32, mpc_ioapic_addr(idx), (*gsi_cfg).gsi_base, (*gsi_cfg).gsi_end) };
0
}

#[unsafe(no_mangle)]
extern "C" fn mp_unregister_ioapic(gsi_base: u32) -> i32 {
let mut __slate_compound_10: pgprot = unsafe { core::mem::zeroed() };
let mut ioapic: i32 = 0;
let mut pin: i32 = 0;
let mut found: i32 = 0;
while ioapic < unsafe { nr_ioapics } {
if (unsafe { ioapics.0[ioapic as usize].gsi_config.gsi_base }) == gsi_base {
found = 1;
break;
}
ioapic = ioapic.wrapping_add(1);
}
if found == 0 {
let mut __v3: i32;
while false {
}
__v3 = unsafe { _printk(c"\x014can't find IOAPIC for GSI %d\n".as_ptr() as *const u8, gsi_base) };
return 19i32.wrapping_neg();
}
while pin < unsafe { ioapics.0[ioapic as usize].nr_registers } {
let gsi: u32 = mp_pin_to_gsi(ioapic, pin);
let irq: i32 = mp_map_gsi_to_irq(gsi, 0u32, core::ptr::null_mut::<irq_alloc_info>());
if irq >= 0 {
let data: *mut mp_chip_data = irq_get_chip_data(irq as u32) as *mut mp_chip_data;
if !data.is_null() && (unsafe { (*data).count }) != 0 {
let mut __v5: i32;
while false {
}
__v5 = unsafe { _printk(c"\x014pin%d on IOAPIC%d is still in use.\n".as_ptr() as *const u8, pin, ioapic) };
return 16i32.wrapping_neg();
}
}
pin = pin.wrapping_add(1);
}
// Mark entry not present
unsafe {
ioapics.0[ioapic as usize].nr_registers = 0;
}
ioapic_destroy_irqdomain(ioapic);
free_ioapic_saved_registers(ioapic);
if !unsafe { ioapics.0[ioapic as usize].iomem_res }.is_null() {
unsafe { release_resource(ioapics.0[ioapic as usize].iomem_res) };
}
__set_fixmap(516i32.wrapping_add(ioapic) as u32, 0u64, unsafe { *{
__slate_compound_10 = pgprot { pgprot: 0 };
core::ptr::addr_of_mut!(__slate_compound_10)
} });
unsafe { memset((core::ptr::addr_of_mut!(ioapics.0) as *mut ioapic).wrapping_offset(ioapic as isize) as *mut (), 0i32, core::mem::size_of::<ioapic>() as u64) };
0
}

#[unsafe(no_mangle)]
extern "C" fn mp_ioapic_registered(gsi_base: u32) -> i32 {
let mut ioapic: i32 = 0;
while ioapic < unsafe { nr_ioapics } {
if (unsafe { ioapics.0[ioapic as usize].gsi_config.gsi_base }) == gsi_base {
return 1;
}
ioapic = ioapic.wrapping_add(1);
}
0
}

fn mp_irqdomain_get_attr(gsi: u32, data: *mut mp_chip_data, info: *mut irq_alloc_info) {
if !info.is_null() && ((unsafe { (*info).__slate_anon_7.ioapic.__slate_bits_0.__get_valid() }) as i32) != 0 {
unsafe {
(*data).is_level = ((*info).__slate_anon_7.ioapic.__slate_bits_0.__get_is_level() as i32) != 0;
}
unsafe {
(*data).active_low = ((*info).__slate_anon_7.ioapic.__slate_bits_0.__get_active_low() as i32) != 0;
}
} else {
if __acpi_get_override_irq(gsi, unsafe { core::ptr::addr_of_mut!((*data).is_level) }, unsafe { core::ptr::addr_of_mut!((*data).active_low) }) < 0 {
// PCI interrupts are always active low level triggered.
unsafe {
(*data).is_level = true;
}
unsafe {
(*data).active_low = true;
}
}
}
}

/// Configure the I/O-APIC specific fields in the routing entry.
///
/// This is important to setup the I/O-APIC specific bits (is_level,
/// active_low, masked) because the underlying parent domain will only
/// provide the routing information and is oblivious of the I/O-APIC
/// specific bits.
///
/// The entry is just preconfigured at this point and not written into the
/// RTE. This happens later during activation which will fill in the actual
/// routing information.
fn mp_preconfigure_entry(data: *mut mp_chip_data) {
let entry: *mut IO_APIC_route_entry = unsafe { core::ptr::addr_of_mut!((*data).entry) };
unsafe { memset(entry as *mut (), 0i32, core::mem::size_of::<IO_APIC_route_entry>() as u64) };
let __t0 = (unsafe { (*data).is_level }) as u64;
unsafe {
let mut __t1 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t1.__set_is_level(__t0);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0), __t1);
}
let __t2 = (unsafe { (*data).active_low }) as u64;
unsafe {
let mut __t3 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t3.__set_active_low(__t2);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0), __t3);
}
// Mask level triggered irqs. Edge triggered irqs are masked
// by the irq core code in case they fire.
let __t4 = (unsafe { (*data).is_level }) as u64;
unsafe {
let mut __t5 = core::ptr::read_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0));
__t5.__set_masked(__t4);
core::ptr::write_unaligned(core::ptr::addr_of_mut!((*entry).__slate_anon_0.__slate_anon_0.__slate_bits_0), __t5);
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.io_apic.mp_irqdomain_alloc")]
extern "C" fn mp_irqdomain_alloc(domain: *mut irq_domain, virq: u32, nr_irqs: u32, arg: *mut ()) -> i32 {
let mut __slate_compound_21: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut __slate_compound_23: __SlateRecord_slab_h_532 = unsafe { core::mem::zeroed() };
let mut _old: core::mem::MaybeUninit<*mut alloc_tag> = core::mem::MaybeUninit::uninit();
let __slate_slot_1: *mut *mut alloc_tag = core::ptr::addr_of_mut!(_old) as *mut *mut alloc_tag;
let mut _res: core::mem::MaybeUninit<*mut ()> = core::mem::MaybeUninit::uninit();
let __slate_slot_2: *mut *mut () = core::ptr::addr_of_mut!(_res) as *mut *mut ();
let mut __obj_size: u64 = unsafe { core::mem::zeroed() };
let mut flags: core::mem::MaybeUninit<u64> = core::mem::MaybeUninit::uninit();
let __slate_slot_7: *mut u64 = core::ptr::addr_of_mut!(flags) as *mut u64;
let mut pin: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_8: *mut i32 = core::ptr::addr_of_mut!(pin) as *mut i32;
let mut ioapic: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_9: *mut i32 = core::ptr::addr_of_mut!(ioapic) as *mut i32;
let mut ret: core::mem::MaybeUninit<i32> = core::mem::MaybeUninit::uninit();
let __slate_slot_10: *mut i32 = core::ptr::addr_of_mut!(ret) as *mut i32;
let mut irq_data: core::mem::MaybeUninit<*mut irq_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_11: *mut *mut irq_data = core::ptr::addr_of_mut!(irq_data) as *mut *mut irq_data;
let mut data: core::mem::MaybeUninit<*mut mp_chip_data> = core::mem::MaybeUninit::uninit();
let __slate_slot_12: *mut *mut mp_chip_data = core::ptr::addr_of_mut!(data) as *mut *mut mp_chip_data;
let mut info: *mut irq_alloc_info = unsafe { core::mem::zeroed() };
unsafe {
info = arg as *mut irq_alloc_info;
let __slate_auto_bytes_2 = __slate_slot_12 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_2.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_3 = __slate_slot_11 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_3.add(0usize), 0u8, 8usize);
let __slate_auto_bytes_4 = __slate_slot_10 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_4.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_5 = __slate_slot_9 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_5.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_6 = __slate_slot_8 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_6.add(0usize), 0u8, 4usize);
let __slate_auto_bytes_7 = __slate_slot_7 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_7.add(0usize), 0u8, 8usize);
if info.is_null() || nr_irqs > 1 {
return 22i32.wrapping_neg();
} else {
*__slate_slot_11 = irq_domain_get_irq_data(domain, virq);
if (*__slate_slot_11).is_null() {
return 22i32.wrapping_neg();
} else {
*__slate_slot_9 = mp_irqdomain_ioapic_idx(domain);
*__slate_slot_8 = (*info).__slate_anon_7.ioapic.pin;
if !irq_resolve_mapping(domain, *__slate_slot_8 as u64).is_null() {
return 17i32.wrapping_neg();
} else {
__obj_size = size_mul(core::mem::size_of::<mp_chip_data>() as u64, 1u64);
let __slate_auto_bytes_1 = __slate_slot_2 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes_1.add(0usize), 0u8, 8usize);
if mem_alloc_profiling_enabled() {
let __slate_auto_bytes = __slate_slot_1 as *mut u8;
core::ptr::write_bytes(__slate_auto_bytes.add(0usize), 0u8, 8usize);
*__slate_slot_1 = core::ptr::null_mut::<alloc_tag>();
*__slate_slot_2 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_21 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_21)
});
alloc_tag_restore!();
} else {
*__slate_slot_2 = _kzalloc_noprof(__obj_size, 3264u32, *{
__slate_compound_23 = __SlateRecord_slab_h_532 {  };
core::ptr::addr_of_mut!(__slate_compound_23)
});
}
*__slate_slot_12 = *__slate_slot_2 as *mut mp_chip_data;
if (*__slate_slot_12).is_null() {
return 12i32.wrapping_neg();
} else {
*__slate_slot_10 = irq_domain_alloc_irqs_parent(domain, virq, nr_irqs, info as *mut ());
if *__slate_slot_10 >= 0 {
INIT_LIST_HEAD(core::ptr::addr_of_mut!((*(*__slate_slot_12)).irq_2_pin));
(*(*__slate_slot_11)).hwirq = (*info).__slate_anon_7.ioapic.pin as u64;
(*(*__slate_slot_11)).chip = if (*domain).parent == x86_vector_domain { core::ptr::addr_of_mut!(ioapic_chip) } else { core::ptr::addr_of_mut!(ioapic_ir_chip) };
(*(*__slate_slot_11)).chip_data = *__slate_slot_12 as *mut ();
mp_irqdomain_get_attr(mp_pin_to_gsi(*__slate_slot_9, *__slate_slot_8), *__slate_slot_12, info);
if !add_pin_to_irq_node(*__slate_slot_12, ioapic_alloc_attr_node(info), *__slate_slot_9, *__slate_slot_8) {
*__slate_slot_10 = 12i32.wrapping_neg();
irq_domain_free_irqs_parent(domain, virq, nr_irqs);
} else {
mp_preconfigure_entry(*__slate_slot_12);
mp_register_handler(virq, (*(*__slate_slot_12)).is_level);
local_irq_save!(*__slate_slot_7);
if virq < (nr_legacy_irqs() as u32) {
unsafe { (*legacy_pic).mask }.unwrap()(virq);
}
local_irq_restore!(*__slate_slot_7);
loop {
if 1 <= apic_verbosity {
while false {
}
_printk(c"\x016IOAPIC[%d]: Preconfigured routing entry (%d-%d -> IRQ %d Level:%i ActiveLow:%i)\n".as_ptr() as *const u8, *__slate_slot_9, mpc_ioapic_id(*__slate_slot_9), *__slate_slot_8, virq, (*(*__slate_slot_12)).is_level as i32, (*(*__slate_slot_12)).active_low as i32);
}
if true {
break;
}
}
return 0;
}
}
kfree(*__slate_slot_12 as *const ());
return *__slate_slot_10;
}
}
}
}
}
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.io_apic.mp_irqdomain_free")]
extern "C" fn mp_irqdomain_free(domain: *mut irq_domain, virq: u32, nr_irqs: u32) {
let mut data: *mut mp_chip_data = unsafe { core::mem::zeroed() };
loop {
if unlikely__2!(nr_irqs != 1) != 0 {
loop {
unsafe {
core::arch::asm!("677: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 677b - ., 3; .popsection\n# {0}", const 677 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n", sym __slate_asm_string_mp_irqdomain_free_0, sym __slate_asm_string_mp_irqdomain_free_1, const 2922 as i32, const 0 as i32, const 16 as u64, options(nostack, att_syntax));
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
let irq_data: *mut irq_data = unsafe { irq_domain_get_irq_data(domain, virq) };
if !irq_data.is_null() && !unsafe { (*irq_data).chip_data }.is_null() {
data = (unsafe { (*irq_data).chip_data }) as *mut mp_chip_data;
__remove_pin_from_irq(data, mp_irqdomain_ioapic_idx(domain), (unsafe { (*irq_data).hwirq }) as i32);
let mut __v5: i64;
let __ret_warn_on: i32 = (list_empty((unsafe { core::ptr::addr_of_mut!((*data).irq_2_pin) }) as *const list_head) == 0) as i32;
if unlikely!(__ret_warn_on) != 0 {
loop {
1i32 | BUGFLAG_TAINT!(9i32);
unsafe {
core::arch::asm!("678: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 678b - ., 3; .popsection\n# {0}", const 678 as i32, options(nostack, att_syntax));
}
loop {
unsafe {
core::arch::asm!("1:\t ud2 \n.pushsection __bug_table,\"aw\"\n\t912: .pushsection .discard.annotate_data, \"M\", @progbits, 8; .long 912b - ., 1; .popsection\n\t2:\n\t\t.long 1b - .\t# bug_entry::bug_addr\n\t.long {0} - .\t# bug_entry::format\n\t.long {1} - .\t# bug_entry::file\n\t.word {2}\t# bug_entry::line\n\t.word {3}\t# bug_entry::flags\n\t.org 2b + {4}\n.popsection\n.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 1b - ., 8; .popsection", sym __slate_asm_string_mp_irqdomain_free_6, sym __slate_asm_string_mp_irqdomain_free_7, const 2927 as i32, const 2305 as i32, const 16 as u64, options(nostack, att_syntax));
}
if true {
break;
}
}
unsafe {
core::arch::asm!("679: nop\n\t.pushsection .discard.annotate_insn, \"M\", @progbits, 8; .long 679b - ., 4; .popsection\n# {0}", const 679 as i32, options(nostack, att_syntax));
}
if true {
break;
}
}
}
__v5 = (__ret_warn_on != 0) as i64;
unsafe { kfree((*irq_data).chip_data as *const ()) };
}
unsafe { irq_domain_free_irqs_top(domain, virq, nr_irqs) };
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.io_apic.mp_irqdomain_activate")]
extern "C" fn mp_irqdomain_activate(domain: *mut irq_domain, irq_data: *mut irq_data, reserve: bool) -> i32 {
let mut __UNIQUE_ID_guard_680: __SlateRecord_spinlock_h_594 = class_raw_spinlock_irqsave_constructor(unsafe { core::ptr::addr_of_mut!(ioapic_lock) });
let __slate_cleanup_ptr_0 = core::ptr::addr_of_mut!(__UNIQUE_ID_guard_680) as *mut _;
let __slate_cleanup_0 = ::slate_support::Guard(&mut || class_raw_spinlock_irqsave_destructor(__slate_cleanup_ptr_0 as *mut _));
let mut __UNIQUE_ID_unlock_681: *mut __SlateRecord_spinlock_h_594 = ((unsafe { core::ptr::addr_of_mut!(ioapic_lock) }) as u64) as *mut __SlateRecord_spinlock_h_594;
let __slate_cleanup_ptr_1 = core::ptr::addr_of_mut!(__UNIQUE_ID_unlock_681) as *mut _;
let __slate_cleanup_1 = ::slate_support::Guard(&mut || __class_raw_spinlock_irqsave_cleanup_ctx(__slate_cleanup_ptr_1 as *mut _));
ioapic_configure_entry(irq_data);
0
}

#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.slate_distinct.io_apic.mp_irqdomain_deactivate")]
extern "C" fn mp_irqdomain_deactivate(domain: *mut irq_domain, irq_data: *mut irq_data) {
// It won't be called for IRQ with multiple IOAPIC pins associated
ioapic_mask_entry(mp_irqdomain_ioapic_idx(domain), (unsafe { (*irq_data).hwirq }) as i32);
}

#[unsafe(no_mangle)]
extern "C" fn mp_irqdomain_ioapic_idx(domain: *mut irq_domain) -> i32 {
((unsafe { (*domain).host_data }) as i64) as i32
}

#[unsafe(no_mangle)]
static mp_ioapic_irqdomain_ops: __SlateConst<irq_domain_ops> = __SlateConst({
let mut __t0: irq_domain_ops = unsafe { core::mem::zeroed() };
__t0.alloc = Some(mp_irqdomain_alloc);
__t0.free = Some(mp_irqdomain_free);
__t0.activate = Some(mp_irqdomain_activate);
__t0.deactivate = Some(mp_irqdomain_deactivate);
__t0
});

static __slate_asm_string___kmalloc_index_1: [u8; 1] = *b"\0";

static __slate_asm_string___kmalloc_index_2: [u8; 50] = *b"/home/takashi/c-corpus/linux/include/linux/slab.h\0";

static __slate_asm_string_alloc_irq_from_domain_5: [u8; 35] = *b"ioapic: unknown irqdomain type %d\n\0";

static __slate_asm_string_alloc_irq_from_domain_6: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

static __slate_asm_string_instrument_atomic_check_alignment_3: [u8; 1] = *b"\0";

static __slate_asm_string_instrument_atomic_check_alignment_4: [u8; 58] = *b"/home/takashi/c-corpus/linux/include/linux/instrumented.h\0";

static __slate_asm_string_io_apic_unique_id_11: [u8; 1] = *b"\0";

static __slate_asm_string_io_apic_unique_id_12: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

static __slate_asm_string_mp_find_ioapic_pin_1: [u8; 1] = *b"\0";

static __slate_asm_string_mp_find_ioapic_pin_2: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

static __slate_asm_string_mp_find_ioapic_pin_6: [u8; 1] = *b"\0";

static __slate_asm_string_mp_find_ioapic_pin_7: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

static __slate_asm_string_mp_irqdomain_free_0: [u8; 1] = *b"\0";

static __slate_asm_string_mp_irqdomain_free_1: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

static __slate_asm_string_mp_irqdomain_free_6: [u8; 1] = *b"\0";

static __slate_asm_string_mp_irqdomain_free_7: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

static __slate_asm_string_setup_IO_APIC_3: [u8; 1] = *b"\0";

static __slate_asm_string_setup_IO_APIC_4: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

static __slate_asm_string_unlock_ExtINT_logic_1: [u8; 1] = *b"\0";

static __slate_asm_string_unlock_ExtINT_logic_2: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

static __slate_asm_string_unlock_ExtINT_logic_6: [u8; 1] = *b"\0";

static __slate_asm_string_unlock_ExtINT_logic_7: [u8; 60] = *b"/home/takashi/c-corpus/linux/arch/x86/kernel/apic/io_apic.c\0";

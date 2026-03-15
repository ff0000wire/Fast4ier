//   fft.cpp - impelementation of class
//   of fast Fourier transform - FFT
//
//   The code is property of LIBROW
//   You can use it on your own
//   When utilizing credit LIBROW site

//   http://www.librow.com/articles/article-10

//   Reworked from original from LIBROW (info above) to Arduino Lib in c++
//   Optimized with precomputed twiddle factors
//   Bit-reversal uses hardware RBIT instruction (ARM) or software fallback

//   Include declaration file
#include <Fast4ier.h>
#include <cstdlib>

// Static member definitions
complex *Fast4::_twiddle = nullptr;
unsigned int Fast4::_max_n = 0;
bool Fast4::_initialized = false;

//   Precompute twiddle factors.
//
//   Memory usage for max_n=8192:
//     twiddle: (8192-1) * 8 bytes = ~64KB
//
//   Only forward twiddles are stored. For IFFT, the conjugate
//   (negate imaginary part) is applied at use time — a single
//   sign flip per factor, much cheaper than computing sin/cos.
void Fast4::init(unsigned int max_n)
{
	if (_initialized) {
		if (_max_n == max_n) return;
		free(_twiddle);
		_twiddle = nullptr;
	}

	_max_n = max_n;

	// max_n == 0 means deinit: free tables and use fallback paths
	if (max_n == 0) {
		_initialized = false;
		return;
	}

	// Allocate twiddle table: total entries = max_n - 1
	// Layout: for stage Step=1,2,4,...,max_n/2 store Step factors
	// starting at offset (Step - 1).
	const unsigned int twiddle_count = max_n - 1;
	_twiddle = (complex *)malloc(twiddle_count * sizeof(complex));

	const FLT pi = 3.14159265358979323846;
	unsigned int offset = 0;
	for (unsigned int Step = 1; Step < max_n; Step <<= 1)
	{
		const FLT delta = -pi / FLT(Step);  // forward direction
		for (unsigned int k = 0; k < Step; ++k)
		{
			FLT angle = delta * k;
			_twiddle[offset + k] = complex(cos(angle), sin(angle));
		}
		offset += Step;
	}

	_initialized = true;
}


//   FORWARD FOURIER TRANSFORM
//     Input  - input data
//     Output - transform result
//     N      - length of both input data and result
bool Fast4::FFT(const complex *const Input, complex *const Output, const unsigned int N)
{
	//   Check input parameters
	if (!Input || !Output || N < 1 || N & (N - 1))
		return false;
	//   Initialize data
	Rearrange(Input, Output, N);
	//   Call FFT implementation
	Perform(Output, N);
	//   Succeeded
	return true;
}

//   FORWARD FOURIER TRANSFORM, INPLACE VERSION
//     Data - both input data and output
//     N    - length of input data
bool Fast4::FFT(complex *const Data, const unsigned int N)
{
	//   Check input parameters
	if (!Data || N < 1 || N & (N - 1))
		return false;
	//   Rearrange
	Rearrange(Data, N);
	//   Call FFT implementation
	Perform(Data, N);
	//   Succeeded
	return true;
}

//   INVERSE FOURIER TRANSFORM
//     Input  - input data
//     Output - transform result
//     N      - length of both input data and result
//     Scale  - if to scale result
bool Fast4::IFFT(const complex *const Input, complex *const Output, const unsigned int N, const bool Scale /* = true */)
{
	//   Check input parameters
	if (!Input || !Output || N < 1 || N & (N - 1))
		return false;
	//   Initialize data
	Rearrange(Input, Output, N);
	//   Call FFT implementation
	Perform(Output, N, true);
	//   Scale if necessary
	if (Scale)
		Fast4::Scale(Output, N);
	//   Succeeded
	return true;
}

//   INVERSE FOURIER TRANSFORM, INPLACE VERSION
//     Data  - both input data and output
//     N     - length of both input data and result
//     Scale - if to scale result
bool Fast4::IFFT(complex *const Data, const unsigned int N, const bool Scale /* = true */)
{
	//   Check input parameters
	if (!Data || N < 1 || N & (N - 1))
		return false;
	//   Rearrange
	Rearrange(Data, N);
	//   Call FFT implementation
	Perform(Data, N, true);
	//   Scale if necessary
	if (Scale)
		Fast4::Scale(Data, N);
	//   Succeeded
	return true;
}

//   Compute bit-reversed index using hardware RBIT or software fallback
static inline unsigned int bitrev(unsigned int i, unsigned int bits)
{
#ifdef BUILD_NATIVE
	// Software bit-reversal for x86/host builds
	unsigned int j = 0;
	for (unsigned int b = 0; b < bits; b++)
	{
		j = (j << 1) | (i & 1);
		i >>= 1;
	}
	return j;
#else
	return __builtin_arm_rbit(i) >> (32 - bits);
#endif
}

//   Out-of-place rearrange using bit-reversal
void Fast4::Rearrange(const complex *const Input, complex *const Output, const unsigned int N)
{
	const unsigned int bits = __builtin_ctz(N);
	for (unsigned int i = 0; i < N; ++i)
		Output[bitrev(i, bits)] = Input[i];
}

//   Inplace rearrange using bit-reversal
void Fast4::Rearrange(complex *const Data, const unsigned int N)
{
	const unsigned int bits = __builtin_ctz(N);
	for (unsigned int i = 0; i < N; ++i)
	{
		unsigned int j = bitrev(i, bits);
		if (j > i)
		{
			const complex Temp(Data[j]);
			Data[j] = Data[i];
			Data[i] = Temp;
		}
	}
}

//   FFT implementation using precomputed twiddle factors
void Fast4::Perform(complex *const Data, const unsigned int N, const bool Inverse /* = false */)
{
	if (_initialized && N <= _max_n)
	{
		// Use precomputed twiddle factors.
		// Twiddle layout: stage with Step has Step entries starting at offset (Step-1).
		// The twiddle values only depend on Step and Group index, not on N,
		// so the same table works for all FFT sizes <= _max_n.
		// For IFFT: conjugate the twiddle factor (negate imaginary part).

		//   Iteration through dyads, quadruples, octads and so on...
		for (unsigned int Step = 1; Step < N; Step <<= 1)
		{
			//   Jump to the next entry of the same transform factor
			const unsigned int Jump = Step << 1;
			const unsigned int tw_offset = Step - 1;

			//   Iteration through groups of different transform factor
			for (unsigned int Group = 0; Group < Step; ++Group)
			{
				complex Factor = _twiddle[tw_offset + Group];
				if (Inverse)
				{
					// conjugate: negate imaginary part
					Factor = Factor.conjugate();
				}

				//   Iteration within group
				for (unsigned int Pair = Group; Pair < N; Pair += Jump)
				{
					//   Match position
					const unsigned int Match = Pair + Step;
					//   Second term of two-point transform
					const complex Product(Factor * Data[Match]);
					//   Transform for fi + pi
					Data[Match] = Data[Pair] - Product;
					//   Transform for fi
					Data[Pair] += Product;
				}
			}
		}
	}
	else
	{
		// Fallback: original algorithm with trigonometric recurrence
		const FLT pi = Inverse ? 3.14159265358979323846 : -3.14159265358979323846;
		//   Iteration through dyads, quadruples, octads and so on...
		for (unsigned int Step = 1; Step < N; Step <<= 1)
		{
			//   Jump to the next entry of the same transform factor
			const unsigned int Jump = Step << 1;
			//   Angle increment
			const FLT delta = pi / FLT(Step);
			//   Auxiliary sin(delta / 2)
			const FLT Sine = sin(delta * .5);
			//   Multiplier for trigonometric recurrence
			const complex Multiplier(-2. * Sine * Sine, sin(delta));
			//   Start value for transform factor, fi = 0
			complex Factor(1.);
			//   Iteration through groups of different transform factor
			for (unsigned int Group = 0; Group < Step; ++Group)
			{
				//   Iteration within group
				for (unsigned int Pair = Group; Pair < N; Pair += Jump)
				{
					//   Match position
					const unsigned int Match = Pair + Step;
					//   Second term of two-point transform
					const complex Product(Factor * Data[Match]);
					//   Transform for fi + pi
					Data[Match] = Data[Pair] - Product;
					//   Transform for fi
					Data[Pair] += Product;
				}
				//   Successive transform factor via trigonometric recurrence
				Factor = Multiplier * Factor + Factor;
			}
		}
	}
}

//   FORWARD FFT using RBIT for bit-reversal (kept for API compatibility)
bool Fast4::FFT_rbit(complex *const Data, const unsigned int N)
{
	return FFT(Data, N);
}

//   INVERSE FFT using RBIT for bit-reversal (kept for API compatibility)
bool Fast4::IFFT_rbit(complex *const Data, const unsigned int N, const bool Scale /* = true */)
{
	return IFFT(Data, N, Scale);
}

//   Scaling of inverse FFT result
void Fast4::Scale(complex *const Data, const unsigned int N)
{
	const FLT Factor = 1. / FLT(N);
	//   Scale all data entries
	for (unsigned int Position = 0; Position < N; ++Position)
		Data[Position] *= Factor;
}

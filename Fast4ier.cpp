//   fft.cpp - impelementation of class
//   of fast Fourier transform - FFT
//
//   The code is property of LIBROW
//   You can use it on your own
//   When utilizing credit LIBROW site

//   http://www.librow.com/articles/article-10

//   Reworked from original from LIBROW (info above) to Arduino Lib in c++
//   Optimized with precomputed twiddle factors and bit-reversal LUT

//   Include declaration file
#include <Fast4ier.h>
#include <cstdlib>

// Static member definitions
complex *Fast4::_twiddle = nullptr;
uint16_t *Fast4::_bitrev = nullptr;
unsigned int Fast4::_max_n = 0;
bool Fast4::_initialized = false;

//   Precompute twiddle factors and bit-reversal table.
//
//   Memory usage for max_n=8192:
//     twiddle: (8192-1) * 8 bytes = ~64KB
//     bitrev:  8192 * 2 bytes     = ~16KB
//     Total: ~80KB
//
//   Only forward twiddles are stored. For IFFT, the conjugate
//   (negate imaginary part) is applied at use time — a single
//   sign flip per factor, much cheaper than computing sin/cos.
void Fast4::init(unsigned int max_n)
{
	if (_initialized) {
		if (_max_n == max_n) return;
		free(_twiddle);
		free(_bitrev);
	}

	_max_n = max_n;

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

	// Allocate and compute bit-reversal LUT (uint16_t saves half the RAM)
	_bitrev = (uint16_t *)malloc(max_n * sizeof(uint16_t));
	unsigned int bits = 0;
	for (unsigned int tmp = max_n; tmp > 1; tmp >>= 1) bits++;

	for (unsigned int i = 0; i < max_n; ++i)
	{
		unsigned int reversed = 0;
		unsigned int val = i;
		for (unsigned int b = 0; b < bits; ++b)
		{
			reversed = (reversed << 1) | (val & 1);
			val >>= 1;
		}
		_bitrev[i] = (uint16_t)reversed;
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

//   Rearrange function using precomputed bit-reversal LUT
void Fast4::Rearrange(const complex *const Input, complex *const Output, const unsigned int N)
{
	if (_initialized && N <= _max_n)
	{
		// _bitrev is computed for _max_n bits. For smaller N, multiply index by
		// shift=_max_n/N to pad low bits with zeros before lookup. After bit
		// reversal those zeros become high bits, producing bitrev_log2(N)(i).
		const unsigned int shift = _max_n / N;
		for (unsigned int i = 0; i < N; ++i)
			Output[_bitrev[i * shift]] = Input[i];
	}
	else
	{
		//   Data entry position
		unsigned int Target = 0;
		//   Process all positions of input signal
		for (unsigned int Position = 0; Position < N; ++Position)
		{
			//  Set data entry
			Output[Target] = Input[Position];
			//   Bit mask
			unsigned int Mask = N;
			//   While bit is set
			while (Target & (Mask >>= 1))
				//   Drop bit
				Target &= ~Mask;
			//   The current bit is 0 - set it
			Target |= Mask;
		}
	}
}

//   Inplace version of rearrange function using precomputed bit-reversal LUT
void Fast4::Rearrange(complex *const Data, const unsigned int N)
{
	if (_initialized && N <= _max_n)
	{
		const unsigned int shift = _max_n / N;
		for (unsigned int i = 0; i < N; ++i)
		{
			unsigned int j = _bitrev[i * shift];
			if (j > i)
			{
				//   Swap entries
				const complex Temp(Data[j]);
				Data[j] = Data[i];
				Data[i] = Temp;
			}
		}
	}
	else
	{
		//   Swap position
		unsigned int Target = 0;
		//   Process all positions of input signal
		for (unsigned int Position = 0; Position < N; ++Position)
		{
			//   Only for not yet swapped entries
			if (Target > Position)
			{
				//   Swap entries
				const complex Temp(Data[Target]);
				Data[Target] = Data[Position];
				Data[Position] = Temp;
			}
			//   Bit mask
			unsigned int Mask = N;
			//   While bit is set
			while (Target & (Mask >>= 1))
				//   Drop bit
				Target &= ~Mask;
			//   The current bit is 0 - set it
			Target |= Mask;
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

//   Scaling of inverse FFT result
void Fast4::Scale(complex *const Data, const unsigned int N)
{
	const FLT Factor = 1. / FLT(N);
	//   Scale all data entries
	for (unsigned int Position = 0; Position < N; ++Position)
		Data[Position] *= Factor;
}

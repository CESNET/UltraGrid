/*
 * =====================================================================================
 *
 *       Filename:  rand_pmms.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  05/31/2012 02:23:23 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *   Organization:  
 *
 * =====================================================================================
 */


#include <cmath>
#include <assert.h>

class Rand_pmms
{
    public:
	Rand_pmms() {};

	void seedi ( unsigned long int s )
	{
            assert(s > 0ul && s < 0x7FFFFFFFul);
            seed = s;
            val = s;
	}

	unsigned long int pmms_rand ( unsigned long int maxv )
	{
	    unsigned long int raw_value = nextrand();

	    return (unsigned long) ((double)maxv * (double)raw_value / (double)0x7FFFFFFF);
	}
    private:

	unsigned long int nextrand()
	{
	    unsigned long long int const a = 16807ull;
	    unsigned long long int const m = 0x7FFFFFFFull;
	    
	    val = (a * val) % m;

	    return val;        
	}

	unsigned long int seed;
        unsigned long int val;
};

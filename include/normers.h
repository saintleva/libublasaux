#ifndef __LIBUBLASAUX_NORMERS_H__
#define __LIBUBLASAUX_NORMERS_H__

/*
 * Copyright (C) Anton Liaukevich 2009 <leva.dev@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/numeric/ublas/functional.h>

namespace boost { namespace numeric { namespace ublas {


template<class Vector>
struct VectorComparer1: public vector_norm_1<Vector>
{

    class NormAccumulator: public vector_scalar_real_unary_functor<V> {
    public:

        inline result_type getCurrent()
        {
            return current;
        }

        inline void accumulate(const value_type& item)
        {
            current += type_traits<value_type>::type_abs(item);
        }

    private:

        real_type current;
    };

/*
    template<class E>
    bool roughlyCompare(const vector_expression<E>& vector1, const vector_expression<E>& vector2)
    {
    }
*/
};


template<
         template<class> class NormAccumulator,
         template<class> class DispatchComparer = DispatchComparer
        >
struct RoughlyVectorComparer: protected StdDispatchComparer {

    template<class E>
    inline static
    bool compare(const vector_expression<E>& vector1, const vector_expression<E>& vector2,
                 typename type_traits<typename E::value_type>::real_type maxDiff)
    {
        doCompare(vector1, vector2, maxDiff)
    }

};


class StdDispatchComparer {

    template<class Container>
    struct Dispatch_ {};

public:

    bool compare(const vector_expression<E>& vector1, const vector_expression<E>& vector2,
                 typename type_traits<typename E::value_type>::real_type maxDiff)
    {
        Dispatch_<E>::compare(vector1, vector2, maxDiff);
    }




private:

    template<class Item, class Storage>
    struct Dispatch_< vector<Item,Storage> > {

        static
        bool compare(const vector_expression<E>& vector1, const vector_expression<E>& vector2,
                     typename type_traits<typename E::value_type>::real_type maxDiff)
        {
            FullRandomizer_::randomizeVector(vect, engine, itemDist);
        }
    };

}; //class StdDispatchComparer


}}} //namespace boost::numeric::ublas

#endif //__LIBUBLASAUX_NORMERS_H__

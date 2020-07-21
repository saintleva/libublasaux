#ifndef __LIBUBLASAUX_STDDISPATCHRANDOMIZER_H__
#define __LIBUBLASAUX_STDDISPATCHRANDOMIZER_H__

/*
 * Copyright (C) Anton Liaukevich 2006-2008 <leva.dev@gmail.com>
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

#include <boost/random/variate_generator.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/symmetric.hpp>
#include <boost/numeric/ublas/hermitian.hpp>
#include <boost/numeric/ublas/banded.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector_of_vector.hpp>

namespace boost { namespace numeric { namespace ublas {


/**
 * "DispatchRandomizer" strategy good implementation. Supports all matrix and vector types (templates)
 * from Boost::numeric::uBLAS library and randomizes them without errors & waste of time. This template
 * implements "Monostate" pattern (only static methods).
 * @author Anton Liaukevich
 * @brief "DispatchRandomizer" strategy good implementation
 * @remark Template parameters are taken from wrapper @see RandomGenerator.
 */
template<
         class Engine_,
         class ItemDistribution_,
         class IndexDistributionCreator_
        >
class StdDispatchRandomizer {
private:
    /* Types */

    typedef Engine_ Engine;
    typedef ItemDistribution_ ItemDist;
    typedef IndexDistributionCreator_ IndexDistCreator;
    typedef typename IndexDistCreator::Distribution IndexDist;

    typedef boost::variate_generator<Engine&, ItemDist> ItemDie;
    typedef boost::variate_generator<Engine&, IndexDist> IndexDie;

    /**
     * Dispatchering class. It is a core of this strategy implementation. It delegates
     * random-generating logics using partial specializations of templates (in compile-time).
     */
    template<class Container>
    struct Dispatch_ {};

public:

    /**
     * Dispatching function callable from RandomGenerator (@see RandomGenerator#operator()).
     * It dispatches randomizing with nested (private) "Dispatch_" class (@see Dispatch_).
     */
    template<class Container>
    inline static void randomize(Container& container, Engine& engine, const ItemDist& itemDist)
    {
        Dispatch_<Container>::randomize(container, engine, itemDist);
    }

protected:
    /**
     * Protected destructor
     * @remark Is necessary in order to prevent deletion of object of parent class (RandomGenerator)
     * as an object of descendant class (StdDispatchRandomizer).
     */
    ~StdDispatchRandomizer() {}

private:

    /*
     * Randomize implementation classes (backends)
     */

    struct FullRandomizer_ {

        template<class Vector>
        static void randomizeVector(Vector& vect, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Vector::size_type Size;
            for (Size i = Size(); i < vect.size(); ++i)
                vect(i) = die();
        }

        template<class Matrix>
        static void randomizeMatrix(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Matrix::size_type Size;
            for (Size i = Size(); i < matr.size1(); ++i)
                for (Size j = Size(); j < matr.size2(); ++j)
                    matr(i, j) = die();
        }
    };

    struct SparseRandomizer_ {

        template<class Vector>
        static void randomizeVector(Vector& vect, Engine& engine, const ItemDist& itemDist)
        {
            vect.clear();

            ItemDie itemDie(engine, itemDist);
            IndexDie indexDie(engine, IndexDistCreator::create(vect.size()));

            typedef typename Vector::size_type Size;
            Size remained = vect.nnz_capacity();
            while (remained > Size())
            {
                Size falled = indexDie();
                if (!vect.find_element(falled))
                {
                    vect.insert_element(falled, itemDie());
                    --remained;
                }
            }
        }

        template<class Matrix>
        static void randomizeMatrix(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            matr.clear();

            ItemDie itemDie(engine, itemDist);
            IndexDie index1Die(engine, IndexDistCreator::create(matr.size1())),
                     index2Die(engine, IndexDistCreator::create(matr.size2()));

            typedef typename Matrix::size_type Size;
            Size remained = matr.nnz_capacity();
            while (remained > Size())
            {
                Size falledIndex1 = index1Die(),
                     falledIndex2 = index2Die();
                if (!matr.find_element(falledIndex1, falledIndex2))
                {
                    matr.insert_element(falledIndex1, falledIndex2, itemDie());
                    --remained;
                }
            }
        }
    };

    struct TriangleRandomizer_ {

        template<class Matrix>
        static void randomizeLower(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Matrix::size_type Size;
            for (Size i = Size(); i < matr.size1(); ++i)
                for (Size j = Size(); j < matr.size2() && j <= i; ++j)
                    matr(i, j) = die();
        }

        template<class Matrix>
        static void randomizeUnitLower(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Matrix::size_type Size;
            for (Size i = Size(); i < matr.size1(); ++i)
                for (Size j = Size(); j < matr.size2() && j < i; ++j)
                    matr(i, j) = die();
        }

        template<class Matrix>
        static void randomizeUpper(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Matrix::size_type Size;
            for (Size i = Size(); i < matr.size1(); ++i)
                for (Size j = i; j < matr.size2(); ++j)
                    matr(i, j) = die();
        }

        template<class Matrix>
        static void randomizeUnitUpper(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Matrix::size_type Size;
            for (Size i = Size(); i < matr.size1(); ++i)
                for (Size j = i + 1; j < matr.size2(); ++j)
                    matr(i, j) = die();
        }
    };

    struct SymmetricRandomizer_ {

        template<class Matrix>
        static void randomize(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Matrix::size_type Size;
            for (Size i = Size(); i < matr.size1(); ++i)
                for (Size j = Size(); j <= i; ++j)
                    matr(i, j) = die();
        }
    };

    struct HermitianRandomizer_ {

        template<class Matrix>
        static void randomize(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Matrix::size_type Size;
            for (Size i = Size(); i < matr.size1(); ++i)
            {
                for (Size j = Size(); j < i; ++j)
                    matr(i, j) = die();
                matr(i, i) = type_traits<typename Matrix::value_type>::real(die());
            }
        }
    };

    class BandedRandomizer_ {
    public:

        template<class Matrix>
        static void randomize(Matrix& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie die(engine, itemDist);
            typedef typename Matrix::size_type Size;
            for (Size i = Size(); i < matr.size1(); ++i)
                for (Size j = startFrom_(i, matr.lower()); j <= i + matr.upper() && j < matr.size2(); ++j)
                    matr(i, j) = die();
        }
    private:

        template<class Size>
        inline static Size startFrom_(const Size& base, const Size& lower)
        {
            return base >= lower ? base - lower : Size();
        }
    };

    /*
     * Partial specializations for simple vector types
     */

    template<class Item, class Storage>
    struct Dispatch_< vector<Item,Storage> > {

        inline static
        void randomize(vector<Item,Storage>& vect, Engine& engine, const ItemDist& itemDist)
        {
            FullRandomizer_::randomizeVector(vect, engine, itemDist);
        }
    };

    template<class Item, std::size_t MAX_SIZE>
    struct Dispatch_< bounded_vector<Item,MAX_SIZE> > {

        inline static
        void randomize(bounded_vector<Item,MAX_SIZE>& vect, Engine& engine, const ItemDist& itemDist)
        {
            FullRandomizer_::randomizeVector(vect, engine, itemDist);
        }
    };

    template<class Item, std::size_t SIZE>
    struct Dispatch_< c_vector<Item,SIZE> > {

        inline static
        void randomize(c_vector<Item,SIZE>& vect, Engine& engine, const ItemDist& itemDist)
        {
            FullRandomizer_::randomizeVector(vect, engine, itemDist);
        }
    };

    template<class Item, class Alloc>
    struct Dispatch_< zero_vector<Item,Alloc> > {

        inline static
        void randomize(zero_vector<Item,Alloc>& vect, Engine& engine, const ItemDist& itemDist) {}
    };

    template<class Item, class Alloc>
    struct Dispatch_< unit_vector<Item,Alloc> > {

        /**
         * @todo Is it right solution or I simply need to leave this specialization empty?
         */
        inline static
        void randomize(unit_vector<Item,Alloc>& vect, Engine& engine, const ItemDist& itemDist)
        {
            typedef unit_vector<Item,Alloc> Vector;
            if (vect.size() > typename Vector::size_type())
            {
                IndexDie indexDie(engine, IndexDistCreator::create(vect.size()));
                Vector temp(vect.size(), indexDie());
                //TODO: Are "noalias" function useful there&
                // noalias(vect) = temp;
                vect = temp;
            }
        }
    };

    template<class Item, class Alloc>
    struct Dispatch_< scalar_vector<Item,Alloc> > {

        /**
         * @todo Is it right solution or I simply need to leave this specialization empty?
         */
        inline static
        void randomize(scalar_vector<Item,Alloc>& vect, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie itemDie(engine, itemDist);
            scalar_vector<Item,Alloc> temp(vect.size(), itemDie());
            //TODO: Are "noalias" function useful there&
            // noalias(vect) = temp;
            vect = temp;
        }
    };

    /*
     * Partial specializations for sparse vector types
     */

    template<class Item, class Storage>
    struct Dispatch_< mapped_vector<Item,Storage> > {

        inline static
        void randomize(mapped_vector<Item,Storage>& vect, Engine& engine, const ItemDist& itemDist)
        {
            SparseRandomizer_::randomizeVector(vect, engine, itemDist);
        }
    };

    template<class Item, std::size_t IB, class IndexArray, class ItemArray>
    struct Dispatch_< compressed_vector<Item,IB,IndexArray,ItemArray> > {

        inline static
        void randomize(compressed_vector<Item,IB,IndexArray,ItemArray>& vect, Engine& engine,
                       const ItemDist& itemDist)
        {
            SparseRandomizer_::randomizeVector(vect, engine, itemDist);
        }
    };

    template<class Item, std::size_t IB, class IndexArray, class ItemArray>
    struct Dispatch_< coordinate_vector<Item,IB,IndexArray,ItemArray> > {

        inline static
        void randomize(coordinate_vector<Item,IB,IndexArray,ItemArray>& vect, Engine& engine,
                       const ItemDist& itemDist)
        {
            SparseRandomizer_::randomizeVector(vect, engine, itemDist);
        }
    };

    /*
     * Partial specializations for simple matrix types
     */

    template<class Item, class Orientation, class Storage>
    struct Dispatch_< matrix<Item,Orientation,Storage> > {

        inline static
        void randomize(matrix<Item,Orientation,Storage>& matr, Engine& engine, const ItemDist& itemDist)
        {
            FullRandomizer_::randomizeMatrix(matr, engine, itemDist);
        }
    };

    template<class Item, std::size_t M, std::size_t N, class Orientation>
    struct Dispatch_< bounded_matrix<Item,M,N,Orientation> > {

        inline static
        void randomize(bounded_matrix<Item,M,N,Orientation>& matr, Engine& engine, const ItemDist& itemDist)
        {
            FullRandomizer_::randomizeMatrix(matr, engine, itemDist);
        }
    };

    template<class Item, std::size_t M, std::size_t N>
    struct Dispatch_< c_matrix<Item,M,N> > {

        inline static
        void randomize(c_matrix<Item,M,N>& matr, Engine& engine, const ItemDist& itemDist)
        {
            FullRandomizer_::randomizeMatrix(matr, engine, itemDist);
        }
    };

    template<class Item, class Orientation, class Storage>
    struct Dispatch_< vector_of_vector<Item,Orientation,Storage> > {

        inline static
        void randomize(vector_of_vector<Item,Orientation,Storage>& matr, Engine& engine,
                                        const ItemDist& itemDist)
        {
            FullRandomizer_::randomizeMatrix(matr, engine, itemDist);
        }
    };

    template<class Item, class Alloc>
    struct Dispatch_< zero_matrix<Item,Alloc> > {

        inline static
        void randomize(zero_matrix<Item,Alloc>& matr, Engine& engine, const ItemDist& itemDist) {}
    };

    template<class Item, class Alloc>
    struct Dispatch_< identity_matrix<Item,Alloc> > {

        inline static
        void randomize(identity_matrix<Item,Alloc>& matr, Engine& engine, const ItemDist& itemDist) {}
    };

    template<class Item, class Alloc>
    struct Dispatch_< scalar_matrix<Item,Alloc> > {

        inline static
        void randomize(scalar_matrix<Item,Alloc>& matr, Engine& engine, const ItemDist& itemDist)
        {
            ItemDie itemDie(engine, itemDist);
            scalar_matrix<Item,Alloc> temp(matr.size1(), matr.size2(), itemDie());
            //TODO: Are "noalias" function useful there&
            // noalias(matr) = temp;
            matr = temp;
        }
    };

    /*
     * Partial specializations for triangular, symmetric, hermitian, banded matrix types & they adaptors
     */

    template<class Item, class Type, class Orientation, class Storage>
    struct Dispatch_< triangular_matrix<Item,Type,Orientation,Storage> > {};

    template<class Item, class Orientation, class Storage>
    struct Dispatch_< triangular_matrix<Item,lower,Orientation,Storage> > {

        inline static
        void randomize(triangular_matrix<Item,lower,Orientation,Storage>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            TriangleRandomizer_::randomizeLower(matr, engine, itemDist);
        }
    };

    template<class Item, class Orientation, class Storage>
    struct Dispatch_< triangular_matrix<Item,unit_lower,Orientation,Storage> > {

        inline static
        void randomize(triangular_matrix<Item,unit_lower,Orientation,Storage>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            TriangleRandomizer_::randomizeUnitLower(matr, engine, itemDist);
        }
    };

    template<class Item, class Orientation, class Storage>
    struct Dispatch_< triangular_matrix<Item,upper,Orientation,Storage> > {

        inline static
        void randomize(triangular_matrix<Item,upper,Orientation,Storage>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            TriangleRandomizer_::randomizeUpper(matr, engine, itemDist);
        }
    };

    /**
     * @bug Boost raises "bad_index" exception for triangular matrices of "unit_upper" type
     * and only for non-square squarte matrices. Even so for triangular adaptors it works Ok.
     */
    template<class Item, class Orientation, class Storage>
    struct Dispatch_< triangular_matrix<Item,unit_upper,Orientation,Storage> > {

        inline static
        void randomize(triangular_matrix<Item,unit_upper,Orientation,Storage>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            TriangleRandomizer_::randomizeUnitUpper(matr, engine, itemDist);
        }
    };

    template<class Matrix, class Type>
    struct Dispatch_< triangular_adaptor<Matrix,Type> > {};

    template<class Matrix>
    struct Dispatch_< triangular_adaptor<Matrix,lower> > {

        inline static
        void randomize(triangular_adaptor<Matrix,lower>& matr, Engine& engine, const ItemDist& itemDist)
        {
            TriangleRandomizer_::randomizeLower(matr, engine, itemDist);
        }
    };

    template<class Matrix>
    struct Dispatch_< triangular_adaptor<Matrix,unit_lower> > {

        inline static
        void randomize(triangular_adaptor<Matrix,unit_lower>& matr, Engine& engine, const ItemDist& itemDist)
        {
            TriangleRandomizer_::randomizeUnitLower(matr, engine, itemDist);
        }
    };

    template<class Matrix>
    struct Dispatch_< triangular_adaptor<Matrix,upper> > {

        inline static
        void randomize(triangular_adaptor<Matrix,upper>& matr, Engine& engine, const ItemDist& itemDist)
        {
            TriangleRandomizer_::randomizeUpper(matr, engine, itemDist);
        }
    };

    template<class Matrix>
    struct Dispatch_< triangular_adaptor<Matrix,unit_upper> > {

        inline static
        void randomize(triangular_adaptor<Matrix,unit_upper>& matr, Engine& engine, const ItemDist& itemDist)
        {
            TriangleRandomizer_::randomizeUnitUpper(matr, engine, itemDist);
        }
    };

    template<class Item, class Type, class Orientation, class Storage>
    struct Dispatch_< symmetric_matrix<Item,Type,Orientation,Storage> > {

        inline static
        void randomize(symmetric_matrix<Item,Type,Orientation,Storage>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            SymmetricRandomizer_::randomize(matr, engine, itemDist);
        }
    };

    template<class Matrix, class Type>
    struct Dispatch_< symmetric_adaptor<Matrix,Type> > {

        inline static
        void randomize(symmetric_adaptor<Matrix,Type>& matr, Engine& engine, const ItemDist& itemDist)
        {
            SymmetricRandomizer_::randomize(matr, engine, itemDist);
        }
    };

    template<class Item, class Type, class Orientation, class Storage>
    struct Dispatch_< hermitian_matrix<Item,Type,Orientation,Storage> > {

        inline static
        void randomize(hermitian_matrix<Item,Type,Orientation,Storage>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            HermitianRandomizer_::randomize(matr, engine, itemDist);
        }
    };

    template<class Matrix, class Type>
    struct Dispatch_< hermitian_adaptor<Matrix,Type> > {

        inline static
        void randomize(hermitian_adaptor<Matrix,Type>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            HermitianRandomizer_::randomize(matr, engine, itemDist);
        }
    };

    template<class Item, class Orientation, class Storage>
    struct Dispatch_< banded_matrix<Item,Orientation,Storage> > {

        inline static
        void randomize(banded_matrix<Item,Orientation,Storage>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            BandedRandomizer_::randomize(matr, engine, itemDist);
        }
    };

    template<class Matrix>
    struct Dispatch_< banded_adaptor<Matrix> > {

        inline static
        void randomize(banded_adaptor<Matrix>& matr, Engine& engine, const ItemDist& itemDist)
        {
            BandedRandomizer_::randomize(matr, engine, itemDist);
        }
    };

    /*
     * Partial specializations for sparse matrix types
     */

    /**
     * @bug nnz_capacity() method of "mapped_matrix" always returns 0 therefore I don't know how many
     * elements I need to random-generate. As a result I have excluded implementation for
     * "mapped_matrix" leaving specialization of the Dispatch_ template for "mapped_matrix" empty.
     * @remark It works Ok on "compressed_matrix" & "coordinate_matrix" matrix types.
     */
    template<class Item, class Orientation, class Storage>
    struct Dispatch_< mapped_matrix<Item,Orientation,Storage> > {};

    template<class Item, class Orientation, std::size_t IB, class IndexArray, class ItemArray>
    struct Dispatch_< compressed_matrix<Item,Orientation,IB,IndexArray,ItemArray> > {

        inline static
        void randomize(compressed_matrix<Item,Orientation,IB,IndexArray,ItemArray>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            SparseRandomizer_::randomizeMatrix(matr, engine, itemDist);
        }
    };

    template<class Item, class Orientation, std::size_t IB, class IndexArray, class ItemArray>
    struct Dispatch_< coordinate_matrix<Item,Orientation,IB,IndexArray,ItemArray> > {

        inline static
        void randomize(coordinate_matrix<Item,Orientation,IB,IndexArray,ItemArray>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            SparseRandomizer_::randomizeMatrix(matr, engine, itemDist);
        }
    };

    /**
     * @warning I couldn't create an object of any specialization of "generalized_vector_of_vector"
     * template therefore it haven't been tested.
     */
    template<class Item, class Orientation, class Storage>
    struct Dispatch_< generalized_vector_of_vector<Item,Orientation,Storage> > {

        inline static
        void randomize(generalized_vector_of_vector<Item,Orientation,Storage>& matr, Engine& engine,
                       const ItemDist& itemDist)
        {
            SparseRandomizer_::randomizeMatrix(matr, engine, itemDist);
        }
    };

}; //template class StdDispatchRandomizer


}}} //namespace boost::numeric::ublas

#endif //__LIBUBLASAUX_STDDISPATCHRANDOMIZER_H__

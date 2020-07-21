#ifndef __LIBUBLASAUX_TYPEREPLACER_H__
#define __LIBUBLASAUX_TYPEREPLACER_H__

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

#include <boost/type_traits/remove_cv.hpp>
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


class TypeReplacer {
private:

    template<class Container, class New>
    struct ReplaceBackend {};

    /* Partial specializations for vectors */

    template<class Item, class Storage, class New>
    struct ReplaceBackend< vector<Item,Storage>, New > {
        typedef vector<New,Storage> Answer;
    };

    template<class Item, std::size_t MAX_SIZE, class New>
    struct ReplaceBackend< bounded_vector<Item,MAX_SIZE>, New > {
        typedef bounded_vector<New,MAX_SIZE> Answer;
    };

    template<class Item, std::size_t SIZE, class New>
    struct ReplaceBackend< c_vector<Item,SIZE>, New > {
        typedef c_vector<New,SIZE> Answer;
    };

    template<class Item, class Alloc, class New>
    struct ReplaceBackend< zero_vector<Item,Alloc>, New > {
        typedef zero_vector<New,Alloc> Answer;
    };

    template<class Item, class Alloc, class New>
    struct ReplaceBackend< unit_vector<Item,Alloc>, New > {
        typedef unit_vector<New,Alloc> Answer;
    };

    template<class Item, class Alloc, class New>
    struct ReplaceBackend< scalar_vector<Item,Alloc>, New > {
        typedef scalar_vector<New,Alloc> Answer;
    };

    template<class Item, class Storage, class New>
    struct ReplaceBackend< mapped_vector<Item,Storage>, New > {
        typedef mapped_vector<New,Storage> Answer;
    };

    template<class Item, std::size_t IB, class IndexArray, class ItemArray, class New>
    struct ReplaceBackend< compressed_vector<Item,IB,IndexArray,ItemArray>, New > {
        typedef compressed_vector<New,IB,IndexArray,ItemArray> Answer;
    };

    template<class Item, std::size_t IB, class IndexArray, class ItemArray, class New>
    struct ReplaceBackend< coordinate_vector<Item,IB,IndexArray,ItemArray>, New > {
        typedef coordinate_vector<New,IB,IndexArray,ItemArray> Answer;
    };

    /* Partial specializations for matrices */

    template<class Item, class Orientation, class Storage, class New>
    struct ReplaceBackend< matrix<Item,Orientation,Storage>, New > {
        typedef matrix<New,Orientation,Storage> Answer;
    };

    template<class Item, std::size_t M, std::size_t N, class Orientation, class New>
    struct ReplaceBackend< bounded_matrix<Item,M,N,Orientation>, New > {
        typedef bounded_matrix<New,M,N,Orientation> Answer;
    };

    template<class Item, std::size_t M, std::size_t N, class New>
    struct ReplaceBackend< c_matrix<Item,M,N>, New > {
        typedef c_matrix<New,M,N> Answer;
    };

    template<class Item, class Orientation, class Storage, class New>
    struct ReplaceBackend< vector_of_vector<Item,Orientation,Storage>, New > {
        typedef vector_of_vector<New,Orientation,Storage> Answer;
    };

    template<class Item, class Alloc, class New>
    struct ReplaceBackend< zero_matrix<Item,Alloc>, New > {
        typedef zero_matrix<New,Alloc> Answer;
    };

    template<class Item, class Alloc, class New>
    struct ReplaceBackend< identity_matrix<Item,Alloc>, New > {
        typedef identity_matrix<New,Alloc> Answer;
    };

    template<class Item, class Alloc, class New>
    struct ReplaceBackend< scalar_matrix<Item,Alloc>, New > {
        typedef scalar_matrix<New,Alloc> Answer;
    };

    template<class Item, class Type, class Orientation, class Storage, class New>
    struct ReplaceBackend< triangular_matrix<Item,Type,Orientation,Storage>, New > {
        typedef triangular_matrix<New,Type,Orientation,Storage> Answer;
    };

    template<class Matrix, class Type, class New>
    struct ReplaceBackend< triangular_adaptor<Matrix,Type>, New > {
        typedef typename ReplaceBackend<Matrix,New>::Answer MatrixAnswer; // recursive!
        typedef triangular_adaptor<MatrixAnswer,Type> Answer;
    };

    template<class Item, class Type, class Orientation, class Storage, class New>
    struct ReplaceBackend< symmetric_matrix<Item,Type,Orientation,Storage>, New > {
        typedef symmetric_matrix<New,Type,Orientation,Storage> Answer;
    };

    template<class Matrix, class Type, class New>
    struct ReplaceBackend< symmetric_adaptor<Matrix,Type>, New > {
        typedef typename ReplaceBackend<Matrix,New>::Answer MatrixAnswer; // recursive!
        typedef symmetric_adaptor<MatrixAnswer,Type> Answer;
    };

    template<class Item, class Type, class Orientation, class Storage, class New>
    struct ReplaceBackend< hermitian_matrix<Item,Type,Orientation,Storage>, New > {
        typedef hermitian_matrix<New,Type,Orientation,Storage> Answer;
    };

    template<class Matrix, class Type, class New>
    struct ReplaceBackend< hermitian_adaptor<Matrix,Type>, New > {
        typedef typename ReplaceBackend<Matrix,New>::Answer MatrixAnswer; // recursive!
        typedef hermitian_adaptor<MatrixAnswer,Type> Answer;
    };

    template<class Item, class Orientation, class Storage, class New>
    struct ReplaceBackend< banded_matrix<Item,Orientation,Storage>, New > {
        typedef banded_matrix<New,Orientation,Storage> Answer;
    };

    template<class Matrix, class New>
    struct ReplaceBackend< banded_adaptor<Matrix>, New > {
        typedef typename ReplaceBackend<Matrix,New>::Answer MatrixAnswer; // recursive!
        typedef banded_adaptor<MatrixAnswer> Answer;
    };

    template<class Item, class Orientation, class Storage, class New>
    struct ReplaceBackend< mapped_matrix<Item,Orientation,Storage>, New > {
        typedef mapped_matrix<New,Orientation,Storage> Answer;
    };

    template<class Item, class Orientation, std::size_t IB, class IndexArray, class ItemArray, class New>
    struct ReplaceBackend< compressed_matrix<Item,Orientation,IB,IndexArray,ItemArray>, New > {
        typedef compressed_matrix<New,Orientation,IB,IndexArray,ItemArray> Answer;
    };

    template<class Item, class Orientation, std::size_t IB, class IndexArray, class ItemArray, class New>
    struct ReplaceBackend< coordinate_matrix<Item,Orientation,IB,IndexArray,ItemArray>, New > {
        typedef coordinate_matrix<New,Orientation,IB,IndexArray,ItemArray> Answer;
    };

    template<class Item, class Orientation, class Storage, class New>
    struct ReplaceBackend< generalized_vector_of_vector<Item,Orientation,Storage>, New > {
        typedef generalized_vector_of_vector<New,Orientation,Storage> Answer;
    };

public:

    template<class Container, class New>
    struct Replace {
        typedef typename ReplaceBackend<typename remove_cv<Container>::type, New>::Answer Answer;
    };

}; //template class TypeReplacer


}}} //namespace boost::numeric::ublas

#endif //__LIBUBLASAUX_TYPEREPLACER_H__

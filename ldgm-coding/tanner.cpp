/*
 * =====================================================================================
 *
 *       Filename:  tanner.cpp
 *
 *    Description:  Implementation of Tanner graph
 *
 *        Version:  1.0
 *        Created:  03/01/2012 11:22:55 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *        Company:  FI MUNI
 *
 * =====================================================================================
 */


#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "tanner.h"

using namespace std;

/*-----------------------------------------------------------------------------
 *  Implementation of class Node
 *-----------------------------------------------------------------------------*/
Node::Node(Tanner_graph *tanner, Node::Node_type t, char *d) {
//    int size = tanner->data_size;
    this->type = t;
    data = d;
    done = false;
//    neighbours[0] = 0;
//    printf("Created node with data: %d\n", *data);
}

Node::~Node() {
}

int Node::setDataPtr(char *d) {
    
    if ( d != 0 ) {
	sprintf(data, d, tanner->data_size);
	return 0;
    } else 
	return 1;
}


/*-----------------------------------------------------------------------------
 *  Implementation fo class Edge
 *-----------------------------------------------------------------------------*/

Edge::Edge(Node *a, Node *b) {
    first = a;
    second = b;
}


/*-----------------------------------------------------------------------------
 *  Implementation fo class Tanner_graph
 *-----------------------------------------------------------------------------*/
void Tanner_graph::set_data_size(int size) {
    data_size = size;
}

void Tanner_graph::add_node(Node::Node_type type, int index, char *data) {
//    printf("Creating node with data: %d\n", *data);

    if(data == NULL) {
//	void *d = calloc(data_size, sizeof(char));
	Node n(this, type, data);
	nodes.insert(std::pair <int, Node>(index, n));
    } else {
	Node n(this, type, data);
	nodes.insert(std::pair <int, Node>(index, n));
    }
}



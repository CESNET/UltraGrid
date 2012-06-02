/*
 * =====================================================================================
 *
 *       Filename:  tanner.h
 *
 *    Description:  Definition of Tanner graph
 *
 *        Version:  1.0
 *        Created:  03/01/2012 11:23:26 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Milan Kabat (), kabat@ics.muni.cz
 *        Company:  FI MUNI
 *
 * =====================================================================================
 */


#include <vector>
#include <map>

#ifndef TANNER_H
#define TANNER_H

class Tanner_graph;


/*
 * =====================================================================================
 *        Class:  Node
 *  Description:  A node in Tanner graph
 * =====================================================================================
 */
class Node
{
    
    public:
	enum Node_type {
	    variable_node,
	    constraint_node
	};				/* ----------  end of enum nodeType  ---------- */

	/* ====================  LIFECYCLE     ======================================= */
	Node ( Tanner_graph *t, Node_type type, char *data);        /* constructor      */
//	Node ( const Node &other );   /* copy constructor */
	~Node ();                           /* destructor       */

	/* ====================  ACCESSORS     ======================================= */

	char* getDataPtr() { return data; }

	Node_type getType() { return type; }

	bool isDone() { return done; }

	/* ====================  MUTATORS      ======================================= */

	int setDataPtr(char *d);
	
	void setType(Node_type t) { type = t; }

	void setDone(bool v) { done = v; }
	
	std::vector <int> neighbours;

	/* ====================  OPERATORS     ======================================= */

//	Node& operator = ( const Node &other ); /* assignment operator */

    protected:
	/* ====================  DATA MEMBERS  ======================================= */
	Tanner_graph *tanner;

    private:
	/* ====================  DATA MEMBERS  ======================================= */
	char *data;
	Node_type type;
	bool done;


}; /* -----  end of class Node  ----- */


/*
 * =====================================================================================
 *        Class:  Edge
 *  Description:  Edge in Tanner graph
 * =====================================================================================
 */
class Edge
{
    public:
	/* ====================  LIFECYCLE     ======================================= */
	Edge (Node *first, Node *second);                             /* constructor      */
	Edge ( const Edge &other );   /* copy constructor */
	~Edge () { }                           /* destructor       */

	/* ====================  ACCESSORS     ======================================= */

	Node* getFirst() { return first; }
	Node* getSecond() { return second; }
	/* ====================  MUTATORS      ======================================= */

	/* ====================  OPERATORS     ======================================= */

	Edge& operator = ( const Edge &other ); /* assignment operator */

    protected:
	/* ====================  DATA MEMBERS  ======================================= */

    private:
	/* ====================  DATA MEMBERS  ======================================= */

	Node *first;
	Node *second;

}; /* -----  end of class Edge  ----- */


/*
 * =====================================================================================
 *        Class:  Tanner_graph
 *  Description:  Tanner graph representation
 * =====================================================================================
 */
class Tanner_graph
{
    public:
	/* ====================  LIFECYCLE     ======================================= */
	Tanner_graph () {}                             /* constructor      */
	Tanner_graph ( const Tanner_graph &other );   /* copy constructor */
	~Tanner_graph () {}                           /* destructor       */

	/* ====================  ACCESSORS     ======================================= */

	int get_data_size() { return data_size; }
//	std::vector<Node>& getNodes() { return nodes; }
//	std::vector<Edge>& getEdges() { return edges; }
	/* ====================  MUTATORS      ======================================= */

	void add_node(Node::Node_type t, int index, char *data);
	void set_data_size(int size);
	/* ====================  OPERATORS     ======================================= */

	Tanner_graph& operator = ( const Tanner_graph &other ); /* assignment operator */
	int data_size;
	std::map<int, Node> nodes;

    protected:
	/* ====================  DATA MEMBERS  ======================================= */

    private:
	/* ====================  DATA MEMBERS  ======================================= */

}; /* -----  end of class Tanner_graph  ----- */

#endif

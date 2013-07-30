/*
 * FILE:   pdb.h - Participant database
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2002 University of Southern California
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * $Revision: 1.2 $
 * $Date: 2009/12/11 15:29:39 $
 *
 */
 
/*
 * A participant database entry. This holds (pointers to) all the
 * information about a particular participant in a teleconference.
 *
 */

/**
 * Deletes decoder for participant when the participant is deleted
 */
typedef void (*decoder_state_deleter_t)(void *);

struct pdb_e {
	uint32_t		 ssrc;
	char			*sdes_cname;
	char			*sdes_name;
	char			*sdes_email;
	char			*sdes_phone;
	char			*sdes_loc;
	char			*sdes_tool;
	char			*sdes_note;
	void                    *decoder_state; ///< state of decoder for participant
        decoder_state_deleter_t  decoder_state_deleter; ///< decoder state deleter
	uint8_t			 pt;	/* Last seen RTP payload type for this participant */
	struct pbuf		*playout_buffer;
	struct tfrc		*tfrc_state;
	struct timeval		 creation_time;	/* Time this entry was created */
};

struct pdb;	/* The participant database */

struct pdb          *pdb_init(void);
void                 pdb_destroy(struct pdb **db);
int                  pdb_add(struct pdb *db, uint32_t ssrc);
struct pdb_e        *pdb_get(struct pdb *db, uint32_t ssrc);

/* Remove the entry indexed by "ssrc" from the database, returning a
 * pointer to it in "item". Returns zero if the entry was present.
 */
int                  pdb_remove(struct pdb *db, uint32_t ssrc, struct pdb_e **item);

typedef void *pdb_iter_t;
/*
 * Iterator for the database.
 */ 
struct pdb_e        *pdb_iter_init(struct pdb *db, pdb_iter_t *it);
struct pdb_e        *pdb_iter_next(pdb_iter_t *it);
void                 pdb_iter_done(pdb_iter_t *it);


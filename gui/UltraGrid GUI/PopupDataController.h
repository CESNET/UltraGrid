//
//  PopupDataController.h
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/4/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <Cocoa/Cocoa.h>
#import "UGController.h"

@interface PopupDataController : NSArrayController {
		UGController *controller;
}

@property (retain) IBOutlet NSMutableArray *displays;
@property (retain) IBOutlet NSComboBoxCell *displayCombo;
@property (retain) IBOutlet NSMutableArray *captures;
@property (retain) IBOutlet NSComboBoxCell *captureCombo;

@property (retain) IBOutlet NSMutableArray *audio_plays;
@property (retain) IBOutlet NSComboBoxCell *audioPlayCombo;
@property (retain) IBOutlet NSMutableArray *audio_caps;
@property (retain) IBOutlet NSComboBoxCell *audioCapCombo;

-(void) awakeFromNib;
-(int) numberOfItemsInComboBoxCell: (NSComboBoxCell *) aComboBox;
-(id) comboBoxCell: (NSComboBoxCell *) aComboBox objectValueForItemAtIndex : (int) index;



@end

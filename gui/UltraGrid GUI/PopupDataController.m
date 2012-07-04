//
//  PopupDataController.m
//  UltraGrid GUI
//
//  Created by Martin Pulec on 11/4/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "PopupDataController.h"


@implementation PopupDataController
@synthesize displays;
@synthesize displayCombo;
@synthesize captures;
@synthesize captureCombo;

@synthesize audio_plays;
@synthesize audioPlayCombo;
@synthesize audio_caps;
@synthesize audioCapCombo;

-(void) awakeFromNib
{
	controller = [[UGController alloc] init];
	
	[displays addObjectsFromArray: [controller getOptionsFromUG:@"-d help"]];
	[captures addObjectsFromArray: [controller getOptionsFromUG:@"-t help"]];
	[audio_caps addObjectsFromArray: [controller getOptionsFromUG:@"-s help"]];
	[audio_plays addObjectsFromArray: [controller getOptionsFromUG:@"-r help"]];
}

-(int) numberOfItemsInComboBoxCell: (NSComboBoxCell *) aComboBox
{
	if(aComboBox == displayCombo) {
		return [displays count];
	}
	if(aComboBox == captureCombo) {
		return [captures count];
	}
	if(aComboBox == audioPlayCombo) {
		return [audio_plays count];
	}
	if(aComboBox == audioCapCombo) {
		return [audio_caps count];
	}
	return 0;
}

-(id) comboBoxCell: (NSComboBoxCell *) aComboBox objectValueForItemAtIndex : (int) index
{
	if(aComboBox == displayCombo) {
		return [displays objectAtIndex: index];
	}
	if(aComboBox == captureCombo) {
		return [captures objectAtIndex: index];
	}
	if(aComboBox == audioPlayCombo) {
		return [audio_plays objectAtIndex: index];
	}
	if(aComboBox == audioCapCombo) {
		return [audio_caps objectAtIndex: index];
	}
	return nil;
}

@end

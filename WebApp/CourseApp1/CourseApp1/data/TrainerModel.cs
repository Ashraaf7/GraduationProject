using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace CourseApp1.data
{
    [MetadataType(typeof(TrainerMetadata))]
    public partial class Trainer
    {
        public static implicit operator Trainer(SelectList v)
        {
            throw new NotImplementedException();
        }
    }
    public class TrainerMetadata
    {
        public int ID { get; set; }
        [Display(Name ="Instructor")]
        public string Name { get; set; }
        public string Email { get; set; }
        public string Description { get; set; }
        public string Website { get; set; }
    }
}